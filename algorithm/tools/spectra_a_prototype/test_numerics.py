"""Synthetic unit fixtures only: no detector accuracy or hardware timing claims."""
import unittest
import torch
from torch.nn import functional as F
from numerics import (fake_quantize,quantize_codes,calibrate_exponent,fake_quantize_weight_groups,
    accumulator_bound,requantize_int,conv2d_int_reference,roi_distortion,guidance_loss)

torch.set_num_threads(2)

class QuantizationTests(unittest.TestCase):
    def test_round_ties_and_symmetric_range(self):
        x=torch.tensor([-9.,-7.5,-2.5,-1.5,-.5,.5,1.5,2.5,7.5,9.])
        self.assertEqual(quantize_codes(x,4,0).tolist(),[-7,-7,-2,-2,0,0,2,2,7,7])
        self.assertEqual(quantize_codes(torch.tensor([-128.,-9.,9.,128.]),8,0).tolist(),[-127,-9,9,127])

    def test_ste_is_clipped_and_exponent_static(self):
        x=torch.tensor([-8.,-7.,-1.,1.,7.,8.],requires_grad=True)
        fake_quantize(x,4,0).sum().backward()
        self.assertEqual(x.grad.tolist(),[0,1,1,1,1,0])

    def test_bad_quantization_inputs(self):
        for bits,e in [(3,0),(4,9),(8,-25),(True,0)]:
            with self.assertRaises(ValueError):quantize_codes(torch.tensor([1.]),bits,e)
        with self.assertRaises(ValueError):fake_quantize(torch.tensor([float('nan')]))

    def test_calibration_ties_and_exact_values(self):
        self.assertEqual(calibrate_exponent(torch.zeros(4),4)['exponent'],8)
        x=torch.tensor([1.,2.,3.])
        result=calibrate_exponent(x,4)
        self.assertEqual(result['exponent'],0);self.assertEqual(result['weighted_squared_error'],0)
        with self.assertRaises(ValueError):calibrate_exponent(torch.ones(2),4,torch.zeros(2))
        with self.assertRaises(ValueError):calibrate_exponent(torch.tensor([1e308],dtype=torch.float64),8)

    def test_output_channel_group_tail(self):
        weight=torch.full((33,1,1,1),1.25)
        out=fake_quantize_weight_groups(weight,[0,-1],4)
        self.assertEqual(out[31,0,0,0],1);self.assertEqual(out[32,0,0,0],1)
        weight[-1]=1.5
        self.assertEqual(fake_quantize_weight_groups(weight,[0,-1],4)[-1,0,0,0],1.5)
        with self.assertRaises(ValueError):fake_quantize_weight_groups(weight,[0],4)
        with self.assertRaises(ValueError):fake_quantize_weight_groups(weight,[0,.5],4)

    def test_accumulator_contract(self):
        self.assertEqual(accumulator_bound(256,3,8,8),256*9*127*127)
        with self.assertRaises(OverflowError):accumulator_bound(10**9,3)

    def test_signed_shift_ties_saturation_and_overflow_guard(self):
        x=torch.tensor([-7,-5,-3,-1,1,3,5,7])
        self.assertEqual(requantize_int(x,-1,8).tolist(),[-4,-2,-2,0,0,2,2,4])
        self.assertEqual(requantize_int(torch.tensor([-1,0,1]),40,4).tolist(),[-7,0,7])
        with self.assertRaises(OverflowError):requantize_int(torch.tensor([2**31]),0)

    def test_requantization_integer_dtype_boundaries(self):
        for dtype in (torch.int8,torch.int16,torch.int32,torch.int64):
            with self.subTest(dtype=dtype):
                self.assertEqual(requantize_int(torch.tensor([-1,0,1,127],dtype=dtype),0).tolist(),[-1,0,1,127])
        self.assertEqual(requantize_int(torch.tensor([0,1,127,255],dtype=torch.uint8),0).tolist(),[0,1,127,127])
        self.assertEqual(requantize_int(torch.tensor([-2**31,2**31-1],dtype=torch.int64),0).tolist(),[-127,127])
        for value in [-2**31-1,2**31]:
            with self.assertRaises(OverflowError):requantize_int(torch.tensor([value],dtype=torch.int64),0)

    def test_integer_conv_padding_stride_multichannel(self):
        torch.manual_seed(17)
        x=torch.randint(-7,8,(2,3,7,9));w=torch.randint(-7,8,(5,3,3,3));b=torch.arange(5)
        for stride in [1,2]:
            actual=conv2d_int_reference(x,w,b,stride,1,4,4)
            reference=F.conv2d(x.double(),w.double(),b.double(),stride=stride,padding=1).to(torch.int64)
            self.assertTrue(torch.equal(actual,reference))
        with self.assertRaises(ValueError):conv2d_int_reference(x*2,w,b,1,1,4,4)
        with self.assertRaises(ValueError):conv2d_int_reference(torch.tensor([[[[-128]]]],dtype=torch.int8),torch.ones((1,1,1,1),dtype=torch.int8))
        uint_result=conv2d_int_reference(torch.tensor([[[[127]]]],dtype=torch.uint8),torch.ones((1,1,1,1),dtype=torch.int64))
        self.assertEqual(int(uint_result.item()),127)

class GuidanceTests(unittest.TestCase):
    def inputs(self,n=3):
        return [torch.zeros(n,requires_grad=True),torch.full((n,2),.5),torch.full((n,2),.25),
                torch.full((n,),.5),torch.full((n,),.7),torch.full((n,),.7),torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool)]

    def test_low_precision_zero_and_mixed_guidance_is_finite(self):
        for dtype in (torch.float16,torch.bfloat16):
            args=[torch.zeros(2,dtype=dtype,requires_grad=True),torch.zeros(2,2,dtype=dtype),
                  torch.zeros(2,2,dtype=dtype),torch.full((2,),.8,dtype=dtype),
                  torch.full((2,),.8,dtype=dtype),torch.full((2,),.8,dtype=dtype),
                  torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool)]
            loss,detail=guidance_loss(*args);self.assertEqual(float(loss.detach()),0.)
            self.assertEqual(loss.dtype,torch.float32);self.assertTrue(bool(torch.isfinite(detail['discrepancy']).all()))
            loss.backward();self.assertTrue(bool(torch.isfinite(args[0].grad).all()))
            args[0].grad=None;args[-2][0]=True;args[-1][0]=True;args[2][0]=.5
            loss,detail=guidance_loss(*args);self.assertTrue(bool(torch.isfinite(loss)))
            self.assertEqual(detail['admitted'].tolist(),[True,False]);loss.backward()
            self.assertTrue(bool(torch.isfinite(args[0].grad).all()))

    def test_empty_guidance_is_zero(self):
        args=self.inputs(0);loss,detail=guidance_loss(*args)
        self.assertEqual(float(loss.detach()),0);loss.backward();self.assertEqual(args[0].grad.numel(),0)

    def test_threshold_inclusive_bounded_weights(self):
        args=self.inputs();loss,d=guidance_loss(*args)
        self.assertTrue(bool(d['admitted'].all()));self.assertTrue(bool((d['weights']<=2*d['reliability']).all()))
        self.assertAlmostEqual(float(loss.detach()),.69314718,places=6)

    def test_class_and_missing_masks_do_not_create_negatives(self):
        args=self.inputs();args[-1][:]=False
        loss,d=guidance_loss(*args);loss.backward()
        self.assertEqual(float(loss.detach()),0);self.assertTrue(torch.equal(args[0].grad,torch.zeros(3)))
        args=self.inputs();args[-2][1]=False;args[3][2]=.49
        loss,d=guidance_loss(*args);self.assertEqual(d['admitted'].tolist(),[True,False,False])

    def test_teacher_and_discrepancy_are_detached(self):
        args=self.inputs();args[1].requires_grad_();args[2].requires_grad_();args[3].requires_grad_()
        loss,d=guidance_loss(*args);loss.backward()
        self.assertIsNone(args[1].grad);self.assertIsNone(args[2].grad);self.assertIsNone(args[3].grad)
        self.assertIsNotNone(args[0].grad)

    def test_bad_probability_and_shapes(self):
        args=self.inputs();args[3][0]=1.1
        with self.assertRaises(ValueError):guidance_loss(*args)
        args=self.inputs();args[2]=torch.ones((2,2))
        with self.assertRaises(ValueError):guidance_loss(*args)

class RegionTests(unittest.TestCase):
    def test_zero_low_precision_and_tiny_epsilon(self):
        for dtype in (torch.float16,torch.bfloat16):
            fp=torch.zeros(1,1,2,2,dtype=dtype);q=fp.clone().requires_grad_()
            loss,detail=roi_distortion(fp,q,torch.empty(0,4),1)
            self.assertEqual(float(loss.detach()),0.);self.assertEqual(loss.dtype,torch.float32)
            loss.backward();self.assertTrue(bool(torch.isfinite(q.grad).all()))
        with self.assertRaises(ValueError):roi_distortion(fp,q,torch.empty(0,4),1,eps=1e-300)

    def test_exact_distortion_background_and_gradient(self):
        fp=torch.ones(1,2,4,4);q=torch.zeros_like(fp,requires_grad=True)
        loss,d=roi_distortion(fp,q,torch.tensor([[0.,0.,1.,1.]]),2)
        self.assertAlmostEqual(float(loss.detach()),1,places=6);self.assertEqual(d['background_cells'],15)
        loss.backward();self.assertTrue(bool(torch.isfinite(q.grad).all()))

    def test_empty_and_invalid_regions_reported(self):
        fp=torch.zeros(1,1,2,2);q=fp.clone()
        loss,d=roi_distortion(fp,q,torch.tensor([[2.,2.,1.,1.],[100.,100.,110.,110.]]),1)
        self.assertEqual(d['rejected_boxes'],2);self.assertEqual(float(loss),0)
        loss,d=roi_distortion(fp,q,torch.empty(0,4),1)
        self.assertEqual(d['background_cells'],4);self.assertEqual(d['used_boxes'],0)

if __name__=='__main__':unittest.main()
