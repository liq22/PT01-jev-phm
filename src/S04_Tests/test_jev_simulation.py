"""Small tests of physical and statistical semantics, not efficacy evidence."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"S01_Package"))
from jev_simulation import orders, impulse_response, generate_unit, extract_features, event_losses, weighted_libra_losses, predict_logits, per_unit, fit_head, tensor_data
CFG=json.loads((Path(__file__).resolve().parents[1]/"S02_Configs/jev_simulation_poc.json").read_text())

class SimulationTests(unittest.TestCase):
    def test_characteristic_orders(self):
        self.assertAlmostEqual(orders(CFG)[0],4.8);self.assertAlmostEqual(orders(CFG)[1],3.2)
    def test_response_decay(self):
        h=impulse_response(16000,3000,0.03)
        self.assertEqual(h[0],0.)
        self.assertLess(np.max(np.abs(h[-100:])),1e-10)
    def test_periodic_envelope(self):
        fs=16000;n=32000;period=200
        from scipy.signal import fftconvolve,hilbert
        events=np.zeros(n+3200);events[::period]=1
        x=fftconvolve(events,impulse_response(fs,3000,.03))[:len(events)][1600:1600+n]
        env=np.abs(hilbert(x))**2;env-=env.mean()
        p=np.abs(np.fft.rfft(env*np.hanning(n)))**2;f=np.fft.rfftfreq(n,1/fs)
        peak=f[np.where((f>60)&(f<100))[0][np.argmax(p[(f>60)&(f<100)])]]
        self.assertAlmostEqual(peak,80.,places=1)
    def test_repeatability_and_shape(self):
        d=CFG['domains']['fine_train']
        a=generate_unit(999,1,0,1,CFG,d,6)[0];b=generate_unit(999,1,0,1,CFG,d,6)[0]
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(4,8192))
    def test_features_no_label_argument(self):
        import inspect
        self.assertEqual(list(inspect.signature(extract_features).parameters),['waves','rates','cfg'])
    def test_event_exact_ce(self):
        z=torch.tensor([[.2,.4,.6]],dtype=torch.float64)
        a=torch.tensor([[False,True,False]])
        self.assertAlmostEqual(event_losses(z,a).item(),torch.nn.functional.cross_entropy(z,torch.tensor([1])).item())
    def test_libra_detached_coefficient_gradient(self):
        z=torch.tensor([[.2,.4,.6]],dtype=torch.float64,requires_grad=True);a=torch.tensor([[False,True,True]])
        p=torch.softmax(z,1);mass=p[0,1:].sum()
        expected=(1-mass).detach()*(2*torch.log((1-mass).clamp_min(1e-5))-torch.log(p[0,1:]+1e-5).sum())
        g1=torch.autograd.grad(expected,z,retain_graph=True)[0]
        g2=torch.autograd.grad(weighted_libra_losses(z,a).sum(),z)[0]
        torch.testing.assert_close(g1,g2)
    def test_frozen_conditionals_finite_update(self):
        x=torch.tensor([[1.,2.,1.],[2.,1.,1.]],dtype=torch.float64)
        init=torch.arange(9,dtype=torch.float64).reshape(3,3)/10
        m=dict(mode='freeze',initial=init,weight=torch.tensor([[1.],[2.],[-.5]],dtype=torch.float64))
        z=predict_logits(m,x);z0=x@init
        torch.testing.assert_close(z[:,1]-z[:,2],z0[:,1]-z0[:,2])
    def test_nll_chain_rule(self):
        p=np.array([[.1,.7,.2],[.3,.2,.5]])
        rr=per_unit(p,np.array([1,2]),np.array(['a','b']),p)
        for r in rr:self.assertAlmostEqual(r['nll'],r['coarse_nll']+r['conditional_nll'])
    def test_phase_control_preserves_magnitude(self):
        x=np.random.default_rng(11).normal(size=8192);f=np.fft.rfft(x)
        phase=np.random.default_rng(12).uniform(-np.pi,np.pi,len(f));phase[0]=np.angle(f[0]);phase[-1]=np.angle(f[-1])
        y=np.fft.irfft(abs(f)*np.exp(1j*phase),n=len(x))
        np.testing.assert_allclose(abs(np.fft.rfft(y)),abs(f),atol=1e-10)

    def test_zero_anchor_same_update(self):
        import copy
        cfg=copy.deepcopy(CFG);cfg['optimizer']['steps']=4;cfg['optimizer']['evaluate_every']=1
        x=torch.tensor([[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]],dtype=torch.float64)
        fine=torch.eye(3,dtype=torch.bool);coarse=fine.clone();coarse[1:]=torch.tensor([False,True,True])
        w=torch.full((3,),1/3,dtype=torch.float64)
        tr=[(x,fine,w),(x,coarse,w)];init=torch.zeros((3,3),dtype=torch.float64)
        a,_=fit_head('event',tr,tr,cfg,.001,initial=init)
        b,_=fit_head('anchor',tr,tr,cfg,.001,initial=init,anchor=0.)
        torch.testing.assert_close(a['weight'],b['weight'],rtol=0,atol=0)
    def test_coarse_training_tensor_ignores_hidden_fine_truth(self):
        from sklearn.preprocessing import StandardScaler
        d={'x':np.array([[0.,1.],[1.,0.],[2.,1.]]),'y':np.array([0,1,2]),
           'allowed':np.array([[1,0,0],[0,1,1],[0,1,1]],dtype=bool),'unit':np.array(['a','b','c'])}
        scaler=StandardScaler().fit(d['x']);a=tensor_data(d,scaler)
        d['y']=np.array([2,0,1]);b=tensor_data(d,scaler)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb)

if __name__=='__main__':unittest.main()
