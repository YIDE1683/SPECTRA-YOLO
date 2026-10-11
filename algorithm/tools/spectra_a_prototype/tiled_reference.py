"""Small integer convolution tile execution reference; not a deployed backend.

Complete tensors below model external storage in the intended schedule. Actual
host RAM usage is not a measurement of the planned on-chip buffers.
"""
import torch
from torch.nn import functional as F
from numerics import accumulator_bound,conv2d_int_reference,INTEGER_DTYPES,INT32_MAX,_bits


def tiled_conv2d_int(x,weight,bias=None,stride=1,padding=0,tile_hw=(16,16),
                     input_block=16,output_block=32,activation_bits=8,weight_bits=8):
    if x.ndim!=4 or weight.ndim!=4 or x.shape[1]!=weight.shape[1] or weight.shape[2]!=weight.shape[3]:raise ValueError('NCHW/OIHW mismatch')
    if x.dtype not in INTEGER_DTYPES or weight.dtype not in INTEGER_DTYPES:raise ValueError('integer codes required')
    if x.device.type!='cpu' or weight.device.type!='cpu':raise ValueError('CPU golden reference only')
    # Validate even locations a strided kernel will not visit. Acceptance must
    # not depend on tile size, cropping, or the particular sampled positions.
    x64=x.to(torch.int64);w64=weight.to(torch.int64)
    qa,qw=_bits(activation_bits),_bits(weight_bits)
    if bool(((x64 < -qa)|(x64 > qa)).any()) or bool(((w64 < -qw)|(w64 > qw)).any()):raise ValueError('input code exceeds declared precision')
    if type(stride) is not int or stride not in (1,2) or type(padding) is not int or padding<0:raise ValueError('stride/padding invalid')
    if not isinstance(tile_hw,(tuple,list)) or len(tile_hw)!=2:raise ValueError('tile_hw must have two dimensions')
    if any(type(v) is not int or v<=0 for v in (*tile_hw,input_block,output_block)):raise ValueError('positive integer tile/block dimensions required')
    n,cin,height,width=x.shape;cout,_,kernel,_=weight.shape
    if min(n,cin,cout,height,width,kernel)<=0:raise ValueError('nonempty tensors required')
    hout=(height+2*padding-kernel)//stride+1;wout=(width+2*padding-kernel)//stride+1
    if min(hout,wout)<=0:raise ValueError('empty output')
    if bias is None:bias=torch.zeros(cout,dtype=torch.int64)
    if bias.dtype not in INTEGER_DTYPES or bias.shape!=(cout,) or bias.device.type!='cpu':raise ValueError('integer per-output bias required')
    b64=bias.to(torch.int64)
    if bool((b64 < -(2**31)).any()) or bool((b64>INT32_MAX).any()):raise OverflowError('bias outside INT32')
    accumulator_bound(cin,kernel,activation_bits,weight_bits,int(b64.abs().max()))
    output=torch.empty((n,cout,hout,wout),dtype=torch.int64)
    output_tiles=0;input_chunks=0
    for o0 in range(0,cout,output_block):
        o1=min(cout,o0+output_block)
        for oy in range(0,hout,tile_hw[0]):
            ey=min(hout,oy+tile_hw[0]);top=oy*stride-padding;bottom=(ey-1)*stride-padding+kernel
            for ox in range(0,wout,tile_hw[1]):
                ex=min(wout,ox+tile_hw[1]);left=ox*stride-padding;right=(ex-1)*stride-padding+kernel
                acc=torch.zeros((n,o1-o0,ey-oy,ex-ox),dtype=torch.int64)
                for c0 in range(0,cin,input_block):
                    c1=min(cin,c0+input_block)
                    y0=max(0,top);y1=min(height,bottom);x0=max(0,left);x1=min(width,right)
                    # Contract kernels have padding <= K//2, ensuring each tile intersects input.
                    if y1<=y0 or x1<=x0:raise ValueError('all-padding tile outside supported contract')
                    patch=x[:,:,y0:y1,x0:x1][:,c0:c1]
                    patch=F.pad(patch,(max(0,-left),max(0,right-width),max(0,-top),max(0,bottom-height)))
                    partial=conv2d_int_reference(patch,weight[o0:o1,c0:c1],None,stride,0,activation_bits,weight_bits)
                    acc+=partial;input_chunks+=1
                acc+=b64[None,o0:o1,None,None]
                output[:,o0:o1,oy:ey,ox:ex]=acc;output_tiles+=1
    return output,{'output_tiles':output_tiles,'input_chunks':input_chunks,
                  'round_between_input_chunks':False,'bias_added_once':True,
                  'scope':'integer convolution equality only, not full detector/FPGA/memory bandwidth'}
