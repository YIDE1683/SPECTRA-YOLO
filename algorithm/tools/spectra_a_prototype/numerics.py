"""Auditable numeric primitives for an UNTRAINED research prototype.

The integer functions are small CPU golden references, not FPGA kernels.
Calibration callers must enforce train-only provenance; this module never reads data.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

INT32_MAX=2**31-1
INTEGER_DTYPES=(torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8)


def _bits(bits):
    if type(bits) is not int or bits not in (4,8):raise ValueError('bits must be 4 or 8')
    return 2**(bits-1)-1


def _exponent(exponent):
    if type(exponent) is not int or not -24<=exponent<=8:
        raise ValueError('exponent must be an integer in [-24,8]')
    return exponent


def _finite_float(x):
    if not isinstance(x,torch.Tensor) or not x.is_floating_point():raise ValueError('floating tensor required')
    if not bool(torch.isfinite(x).all()):raise ValueError('nonfinite tensor')


def _loss_dtype(eps,*tensors):
    # Loss reductions need at least FP32: the default epsilon is zero in FP16.
    dtype=torch.float64 if any(t.dtype==torch.float64 for t in tensors) else torch.float32
    if not math.isfinite(eps) or eps<torch.finfo(dtype).tiny:
        raise ValueError('epsilon must be finite and at least the loss dtype normal minimum')
    return dtype


class _FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bits,exponent):
        limit=_bits(bits);scale=math.ldexp(1.0,_exponent(exponent))
        _finite_float(x)
        ctx.save_for_backward((x>=-limit*scale)&(x<=limit*scale))
        return torch.round(x/scale).clamp(-limit,limit)*scale

    @staticmethod
    def backward(ctx,gradient):
        (mask,)=ctx.saved_tensors
        return gradient*mask,None,None


def fake_quantize(x,bits=8,exponent=-7):
    """Ties-to-even; clipped STE with frozen integer exponent."""
    return _FakeQuant.apply(x,bits,exponent)


def quantize_codes(x,bits=8,exponent=-7):
    _finite_float(x);limit=_bits(bits);scale=math.ldexp(1.0,_exponent(exponent))
    return torch.round(x.detach().double()/scale).clamp(-limit,limit).to(torch.int64)


@torch.no_grad()
def calibrate_exponent(x,bits=8,importance=None):
    """Exact finite search. Zero energy/ties choose the larger exponent."""
    _finite_float(x);_bits(bits)
    if x.numel()==0:raise ValueError('empty calibration tensor')
    values=x.detach().double()
    if importance is None:importance=torch.ones_like(values)
    else:
        _finite_float(importance)
        if importance.shape!=x.shape or bool((importance<0).any()) or not bool((importance>0).any()):
            raise ValueError('importance must be same-shape nonnegative with positive mass')
        importance=importance.detach().double()
    best=None
    for exponent in range(-24,9):
        quant=quantize_codes(values,bits,exponent).double()*math.ldexp(1.0,exponent)
        error=float((importance*(values-quant).square()).sum())
        if not math.isfinite(error):raise ValueError('calibration error overflow; rescale or reject the calibration values')
        candidate=(error,-exponent)
        if best is None or candidate<best[0]:best=(candidate,exponent,error)
    return {'exponent':best[1],'weighted_squared_error':best[2],
            'source_boundary':'caller must supply training-only calibration data'}


def fake_quantize_weight_groups(weight,exponents,bits=8,group_size=32):
    _finite_float(weight);_bits(bits)
    if weight.ndim!=4 or type(group_size) is not int or group_size<=0:raise ValueError('OIHW weight and positive group size required')
    groups=math.ceil(weight.shape[0]/group_size)
    if len(exponents)!=groups:raise ValueError('one exponent required per output-channel group')
    return torch.cat([fake_quantize(weight[start:start+group_size],bits,_exponent(exponents[index]))
                      for index,start in enumerate(range(0,weight.shape[0],group_size))],dim=0)


def accumulator_bound(in_channels,kernel_size,activation_bits=8,weight_bits=8,bias_abs_max=0):
    if any(type(v) is not int or v<=0 for v in (in_channels,kernel_size)):raise ValueError('positive integer dimensions required')
    if type(bias_abs_max) is not int or bias_abs_max<0:raise ValueError('nonnegative integer bias bound required')
    bound=in_channels*kernel_size*kernel_size*_bits(activation_bits)*_bits(weight_bits)+bias_abs_max
    if bound>INT32_MAX:raise OverflowError('declared INT32 accumulator cannot hold the worst-case sum')
    return bound


def _round_shift(value,shift):
    # Python integers avoid overflow on large positive shifts before saturation.
    if shift>=0:return value<<shift
    magnitude=abs(value);divisor=1<<(-shift)
    quotient,remainder=divmod(magnitude,divisor)
    if 2*remainder>divisor or (2*remainder==divisor and quotient%2):quotient+=1
    return quotient if value>=0 else -quotient


def requantize_int(accumulator,shift,output_bits=8):
    """Scalar shift = input_e + weight_e - output_e, ties-to-even, saturation."""
    if not isinstance(accumulator,torch.Tensor) or accumulator.dtype not in INTEGER_DTYPES:raise ValueError('integer accumulator required')
    if accumulator.device.type!='cpu':raise ValueError('golden integer reference is CPU-only')
    if type(shift) is not int or not -56<=shift<=40:raise ValueError('shift outside declared exponent contract')
    # Promote before comparison: PyTorch casts scalar bounds to a narrow integer
    # dtype, which can make an in-range int8/int16 value look out of INT32 range.
    wide=accumulator.to(torch.int64)
    if bool((wide<-(2**31)).any()) or bool((wide>INT32_MAX).any()):raise OverflowError('accumulator outside INT32')
    limit=_bits(output_bits)
    result=[max(-limit,min(limit,_round_shift(int(v),shift))) for v in wide.reshape(-1).tolist()]
    return torch.tensor(result,dtype=torch.int64).reshape(accumulator.shape)


def conv2d_int_reference(x,weight,bias=None,stride=1,padding=0,activation_bits=8,weight_bits=8):
    """Small int64 emulation with a pre-operation INT32 safety gate, no groups."""
    if x.dtype not in INTEGER_DTYPES or weight.dtype not in INTEGER_DTYPES:raise ValueError('integer codes required')
    if x.device.type!='cpu' or weight.device.type!='cpu':raise ValueError('CPU golden reference only')
    if x.ndim!=4 or weight.ndim!=4 or x.shape[1]!=weight.shape[1] or weight.shape[2]!=weight.shape[3]:raise ValueError('incompatible NCHW/OIHW tensors')
    if type(stride) is not int or stride not in (1,2) or type(padding) is not int or padding<0:raise ValueError('invalid stride/padding')
    x64=x.to(torch.int64);w64=weight.to(torch.int64)
    qa,qw=_bits(activation_bits),_bits(weight_bits)
    if bool(((x64 < -qa)|(x64 > qa)).any()) or bool(((w64 < -qw)|(w64 > qw)).any()):raise ValueError('input code exceeds declared precision')
    out_channels,cin,kernel,_=weight.shape
    if bias is None:bias=torch.zeros(out_channels,dtype=torch.int64)
    if bias.dtype not in INTEGER_DTYPES or bias.shape!=(out_channels,) or bias.device.type!='cpu':raise ValueError('integer bias per output required')
    if bool((bias.to(torch.int64)<-(2**31)).any()) or bool((bias.to(torch.int64)>INT32_MAX).any()):raise OverflowError('bias outside INT32')
    accumulator_bound(cin,kernel,activation_bits,weight_bits,int(bias.to(torch.int64).abs().max()))
    padded=F.pad(x.to(torch.int64),(padding,padding,padding,padding))
    if padded.shape[2]<kernel or padded.shape[3]<kernel:raise ValueError('kernel exceeds padded input')
    windows=padded.unfold(2,kernel,stride).unfold(3,kernel,stride)
    result=torch.einsum('nchwij,ocij->nohw',windows,weight.to(torch.int64))+bias[None,:,None,None]
    assert bool((result>=-(2**31)).all()) and bool((result<=INT32_MAX).all())
    return result


def roi_distortion(fp,quantized,boxes_xyxy,stride,eps=1e-8):
    """Eq6: summed pixel energies, overlapping boxes counted separately.

    FP values and importance are detached targets; gradient flows through quantized.
    Invalid/empty boxes are excluded with a returned count, never silently repaired.
    """
    _finite_float(fp);_finite_float(quantized);_finite_float(boxes_xyxy)
    dtype=_loss_dtype(eps,fp,quantized)
    if fp.shape!=quantized.shape or fp.ndim!=4 or fp.shape[0]!=1 or min(fp.shape)<=0:raise ValueError('one nonempty NCHW image with matching tensors required')
    if boxes_xyxy.ndim!=2 or boxes_xyxy.shape[1]!=4 or type(stride) is not int or stride<=0:raise ValueError('Nx4 boxes and positive integer stride required')
    target=fp.detach().to(dtype);prediction=quantized.to(dtype)
    error=(target-prediction).square();energy=target.square()
    height,width=fp.shape[2:];union=torch.zeros((height,width),dtype=torch.bool,device=fp.device)
    numerator=(prediction*0).sum();denominator=target.new_tensor(0.);used=0;rejected=0
    for raw in boxes_xyxy.detach().cpu().tolist():
        x1,y1,x2,y2=raw
        if x2<=x1 or y2<=y1:rejected+=1;continue
        lx=max(0,min(width,math.floor(x1/stride)));ly=max(0,min(height,math.floor(y1/stride)))
        rx=max(0,min(width,math.ceil(x2/stride)));ry=max(0,min(height,math.ceil(y2/stride)))
        if rx<=lx or ry<=ly:rejected+=1;continue
        weight=min(4.,max(1.,1024/max((x2-x1)*(y2-y1),1.)))
        numerator=numerator+weight*error[:,:,ly:ry,lx:rx].sum()
        denominator=denominator+weight*energy[:,:,ly:ry,lx:rx].sum()
        union[ly:ry,lx:rx]=True;used+=1
    background=(~union)[None,None,:,:]
    numerator=numerator+.1*(error*background).sum()
    denominator=denominator+.1*(energy*background).sum()
    loss=numerator/(denominator+eps)
    if not bool(torch.isfinite(loss)):raise ValueError('region loss arithmetic overflow')
    return loss,{'used_boxes':used,'rejected_boxes':rejected,'background_cells':int((~union).sum()),'overlap_rule':'count each foreground region separately'}


def guidance_loss(student_logits,fp_scores,q_scores,teacher_scores,gt_iou,flip_iou,class_compatible,matched,eps=1e-8):
    """Matched concept-score consistency; caller supplies train-only one-to-one matches."""
    for tensor in (student_logits,fp_scores,q_scores,teacher_scores,gt_iou,flip_iou):_finite_float(tensor)
    count=student_logits.numel()
    dtype=_loss_dtype(eps,student_logits,fp_scores,q_scores,teacher_scores,gt_iou,flip_iou)
    if student_logits.ndim!=1 or fp_scores.ndim!=2 or fp_scores.shape!=q_scores.shape or fp_scores.shape[0]!=count or fp_scores.shape[1]==0:raise ValueError('unaligned student tensors')
    if any(t.shape!=(count,) for t in (teacher_scores,gt_iou,flip_iou,class_compatible,matched)):raise ValueError('unaligned match tensors')
    if class_compatible.dtype!=torch.bool or matched.dtype!=torch.bool:raise ValueError('explicit boolean class/match masks required')
    for t in (fp_scores,q_scores,teacher_scores,gt_iou,flip_iou):
        if bool(((t<0)|(t>1)).any()):raise ValueError('scores and IoU must be in [0,1]')
    target=teacher_scores.detach().to(dtype);gt=gt_iou.detach().to(dtype);flip=flip_iou.detach().to(dtype)
    fp=fp_scores.detach().to(dtype);quant=q_scores.detach().to(dtype);logits=student_logits.to(dtype)
    gate=matched & class_compatible & (target>=.5) & (gt>=.7) & (flip>=.7)
    discrepancy=((fp-quant).square().mean(dim=1)/(fp.square().mean(dim=1)+eps))
    reliability=gate.to(dtype)*target*flip
    weights=(reliability*(1+discrepancy).clamp(max=2)).detach()
    if count==0 or not bool(gate.any()):loss=(logits*0).sum()
    else:loss=(weights*F.binary_cross_entropy_with_logits(logits,target,reduction='none')).sum()/(weights.sum()+eps)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(discrepancy).all()):raise ValueError('guidance loss arithmetic overflow')
    return loss,{'weights':weights,'admitted':gate,'discrepancy':discrepancy,'reliability':reliability}
