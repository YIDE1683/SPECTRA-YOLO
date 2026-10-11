import unittest
import torch
from numerics import conv2d_int_reference,requantize_int
from tiled_reference import tiled_conv2d_int

class TiledReferenceTests(unittest.TestCase):
    def test_unsampled_invalid_input_is_always_rejected(self):
        x=torch.zeros(1,1,3,3,dtype=torch.int64);x[...,1,1]=128
        weight=torch.ones(1,1,1,1,dtype=torch.int64)
        for tile in ((1,1),(16,16)):
            with self.assertRaises(ValueError):tiled_conv2d_int(x,weight,stride=2,tile_hw=tile)

    def test_multichannel_output_tail_halo_and_rectangles(self):
        torch.manual_seed(29)
        x=torch.randint(-7,8,(2,5,9,11));w=torch.randint(-7,8,(7,5,3,3));b=torch.arange(-3,4)
        for stride in [1,2]:
            for tile in [(2,3),(8,8),(16,16)]:
                out,trace=tiled_conv2d_int(x,w,b,stride,1,tile,2,3,4,4)
                ref=conv2d_int_reference(x,w,b,stride,1,4,4)
                self.assertTrue(torch.equal(out,ref))
                self.assertFalse(trace['round_between_input_chunks'])
                self.assertTrue(trace['bias_added_once'])

    def test_eight_bit_preserves_values_outside_four_bit(self):
        x=torch.tensor([[[[9,127],[-127,-9]],[[20,-20],[40,-40]]]])
        w=torch.tensor([[[[2]],[[3]]]]);bias=torch.tensor([17])
        out,_=tiled_conv2d_int(x,w,bias,1,0,(1,1),1,1,8,8)
        expected=2*x[:,0:1]+3*x[:,1:2]+17
        self.assertTrue(torch.equal(out,expected))

    def test_requantize_once_after_full_accumulation(self):
        x=torch.ones(1,3,1,1,dtype=torch.int64);w=torch.ones(1,3,1,1,dtype=torch.int64)
        out,_=tiled_conv2d_int(x,w,None,1,0,(1,1),1,1,4,4)
        self.assertEqual(int(requantize_int(out,-1,4).item()),2)
        # Rounding each partial1/2 would incorrectly yield0 instead of2.
        self.assertNotEqual(int(requantize_int(out,-1,4).item()),0)

    def test_invalid_contracts_fail_closed(self):
        x=torch.ones(1,1,3,3,dtype=torch.int64);w=x.clone()
        for tile in [(0,1),(1.5,2)]:
            with self.assertRaises(ValueError):tiled_conv2d_int(x,w,tile_hw=tile)
        with self.assertRaises(ValueError):tiled_conv2d_int(x.float(),w)
        with self.assertRaises(ValueError):tiled_conv2d_int(x*127,w,activation_bits=4)

if __name__=='__main__':unittest.main()
