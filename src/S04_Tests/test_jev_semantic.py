"""Isolated numerical/contract tests. Tensor fixtures are NOT PHM experiments."""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src/S01_Package"))
from jev_semantic import PatchAdapter, SignalDecisionModel, event_nll, masked_mean
from jev_learning import record_unit_weights, score_logits, source_scale, train_neural, infer_logits
from jev_learning import infer_semantic_variant
from jev_signal_data import common_rate_signal
from jev_execution import add_subparser, read_config, check_inputs, bind_gpu, release_target

class TinyDecoder(nn.Module):
    """Fixture for differentiable frozen-encoder behavior, not a Qwen substitute."""
    def __init__(self):
        super().__init__(); self.map=nn.Linear(4,4); self.dropout=nn.Dropout(.5)
    def forward(self,inputs_embeds,attention_mask,use_cache,return_dict):
        return SimpleNamespace(last_hidden_state=self.dropout(torch.tanh(self.map(inputs_embeds))))


def make_model(readout="semantic"):
    torch.manual_seed(9)
    return SignalDecisionModel(TinyDecoder(),hidden_size=4,classes=3,patch_size=2,
        readout=readout,candidate_vectors=torch.tensor([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0]]) if readout=="semantic" else None)


class SemanticTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)
    def test_patch_shape_and_no_silent_crop(self):
        p=PatchAdapter(128,4)
        self.assertEqual(p(torch.ones(2,8192,1)).shape,(2,64,4))
        with self.assertRaises(ValueError): p(torch.ones(2,8191,1))
    def test_frozen_backbone_does_not_block_adapter_gradients(self):
        m=make_model();m.train()
        m(torch.arange(16).reshape(2,8,1).float()/16).square().sum().backward()
        self.assertFalse(m.decoder.training)
        self.assertTrue(all(p.grad is None for p in m.decoder.parameters()))
        self.assertTrue(any(p.grad is not None and torch.any(p.grad!=0) for p in m.adapter.parameters()))
    def test_semantic_permutation_reorders_probabilities(self):
        m=make_model();n=copy.deepcopy(m)
        n.candidates.copy_(m.candidates[[1,2,0]])
        x=torch.randn(3,8,1)
        torch.testing.assert_close(n(x).softmax(1),m(x).softmax(1)[:,[1,2,0]])
    def test_declared_variant_does_not_change_checkpoint_candidates(self):
        model=make_model();original=model.candidates.clone();x=torch.randn(3,8,1)
        torch.testing.assert_close(model(x,candidate_vectors=original[[1,2,0]]),model(x)[:,[1,2,0]])
        torch.testing.assert_close(model.adaptation_state()["candidates"],original)
        with self.assertRaisesRegex(ValueError,"same diagnostic events"):
            model(x,candidate_vectors=torch.ones(2,4))
        with self.assertRaisesRegex(ValueError,"duplicates"):
            model(x,candidate_vectors=torch.ones(3,4))
    def test_fixed_source_variant_vectors_are_reused_without_text_encoding(self):
        model=make_model();x=np.arange(72,dtype=np.float32).reshape(9,8,1)/72
        scale={"mean":0.,"std":1.};device=torch.device("cpu")
        with patch("jev_semantic.compile_semantic_variant",side_effect=AssertionError("must stay source-frozen")):
            z,seconds=infer_semantic_variant(model,x,scale,device,3,["H","I","O"],"unused",candidate_vectors=model.candidates.clone())
        np.testing.assert_allclose(z,infer_logits(model,x,scale,device,3))
        self.assertEqual(seconds,0.)
    def test_duplicate_candidate_rejected(self):
        with self.assertRaises(ValueError):
            SignalDecisionModel(TinyDecoder(),hidden_size=4,classes=3,patch_size=2,readout="semantic",candidate_vectors=torch.ones(3,4))
    def test_event_likelihood_exact_and_all(self):
        z=torch.tensor([[.5,1.,-1.],[0.,2.,.3]],requires_grad=True);y=torch.tensor([1,2])
        allowed=torch.nn.functional.one_hot(y,3).bool()
        torch.testing.assert_close(event_nll(z,allowed),torch.nn.functional.cross_entropy(z,y,reduction="none"))
        torch.testing.assert_close(event_nll(z,torch.ones_like(allowed)),torch.zeros(2))
        with self.assertRaises(ValueError): event_nll(z,torch.zeros_like(allowed))
    def test_masked_pooling(self):
        x=torch.tensor([[[1.,3.],[99.,99.]]]);mask=torch.tensor([[1,0]])
        torch.testing.assert_close(masked_mean(x,mask),torch.tensor([[1.,3.]]))
        with self.assertRaises(ValueError): masked_mean(x,mask*0)
    def test_adaptation_roundtrip(self):
        m=make_model();x=torch.randn(3,8,1)
        torch.optim.SGD(m.adapter.parameters(),lr=.1).zero_grad()
        m(x).sum().backward()
        with torch.no_grad():
            for p in m.adapter.parameters(): p.add_(p.grad,alpha=-.1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"adapter.pt";torch.save(m.adaptation_state(),path)
            n=make_model();n.restore_adaptation(torch.load(path,weights_only=True))
            torch.testing.assert_close(m(x),n(x))
            self.assertLess(path.stat().st_size,10000)
    def test_checkpoint_contract_not_silently_changed(self):
        m=make_model();state=m.adaptation_state();state["tau"]=3
        with self.assertRaises(ValueError): m.restore_adaptation(state)
    def test_indexed_readout_shape(self):
        self.assertEqual(make_model("indexed")(torch.randn(2,8,1)).shape,(2,3))
    def test_token_readout_requires_bound_context(self):
        with self.assertRaises(ValueError): make_model("token")
    def test_record_within_unit_weights(self):
        u=np.array(["a","a","a","b"]);r=np.array(["a1","a2","a2","b1"])
        np.testing.assert_allclose(record_unit_weights(u,r),[.25,.125,.125,.5])
        with self.assertRaises(ValueError): record_unit_weights(["a","b"],["same","same"])
    def test_scaler_uses_supplied_source_weights(self):
        x=np.array([[[0.],[2.]],[[4.],[6.]]])
        s=source_scale(x,np.array([.5,.5]));self.assertEqual(s["mean"],3.)
        self.assertAlmostEqual(s["std"],np.sqrt(5.))
    def test_metrics_check_and_reference(self):
        result=score_logits(np.zeros((3,3)),np.array([0,1,2]),np.ones(3)/3)
        self.assertAlmostEqual(result["nll"],np.log(3))
        self.assertAlmostEqual(result["brier"],2/3)
    def test_filter_properties_not_fault_simulation(self):
        t=np.arange(64000)/64000
        x=np.sin(2*np.pi*4000*t)+np.sin(2*np.pi*18000*t)
        y=common_rate_signal(x,sample_rate_hz=64000,channel_index=0,scale_multiplier=1.,target_rate_hz=25600,beta=5.)
        self.assertEqual(y.shape,(25600,1))
        ideal=np.sin(2*np.pi*4000*np.arange(25600)/25600)
        self.assertLess(np.std(y[200:-200,0]-ideal[200:-200]),.01)
        with self.assertRaises(ValueError): common_rate_signal(x,sample_rate_hz=64000.1,channel_index=0,scale_multiplier=1.,target_rate_hz=25600,beta=5.)
    def test_contract_parse(self):
        parser=argparse.ArgumentParser();sub=parser.add_subparsers(required=True)
        add_subparser(sub)
        a=parser.parse_args(["semantic","check","--config","x.yaml","--run-root","x"])
        self.assertEqual(a.stage,"check")
    def test_six_real_configs_are_parsed(self):
        for name in ("rf","signal","indexed","semantic","permuted","token"):
            cfg=read_config(ROOT/f"src/S02_Configs/real/{name}.yaml")
            self.assertEqual(cfg["name"],name)
            self.assertTrue((ROOT / cfg["data"]["qualification_file"]).is_file())
            cfg["data"]["qualification_file"] = None
            with self.assertRaisesRegex(RuntimeError,"BLOCKED"):
                check_inputs(cfg,ROOT,gpu=False)
    def test_forbidden_gpu_rejected_before_cuda_initialization(self):
        with patch("jev_execution.subprocess.check_output",return_value="2, GPU-test\n"):
            with self.assertRaisesRegex(ValueError,"forbidden"):
                bind_gpu({"gpu_uuid":"GPU-test","forbidden_physical_gpu":2})
    def test_unknown_nested_config_is_rejected(self):
        import yaml
        cfg=read_config(ROOT/"src/S02_Configs/real/semantic.yaml")
        cfg["model"]["unconsumed_option"]=True
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"bad.yaml";f.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError,"model parameters"):
                read_config(f)
    def test_probability_floor_does_not_change_brier(self):
        from jev_learning import score_probabilities
        p=np.array([[0.,1.,0.],[.2,.3,.5]])
        y=np.array([0,2]);w=np.ones(2)/2
        a=score_probabilities(p,y,w,floor=1e-12)
        b=score_probabilities(p,y,w,floor=1e-4)
        self.assertEqual(a["brier"],b["brier"])
        self.assertEqual(a["error"],b["error"])
        self.assertEqual(a["log_floor_weight"],.5)
        self.assertGreater(a["nll"],b["nll"])
    def test_target_release_requires_all_source_fits(self):
        cfg=read_config(ROOT/"src/S02_Configs/real/semantic.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,"sealed"):
                release_target(cfg,ROOT,Path(tmp))
    def test_local_learning_save_reload_evaluate_fixture(self):
        """Code-level pipeline only; no real data, pretrained weights or GPU."""
        x=np.arange(72,dtype=np.float32).reshape(9,8,1)/72
        y=np.arange(9)%3;meta={"y":y,"unit":np.array([str(i) for i in range(9)]),"record":np.array([str(i) for i in range(9)])}
        data=(x,meta,np.ones(9)/9);scale=source_scale(x,data[2]);m=make_model()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            result=train_neural(m,data,data,scale,{"batch_size":3,"max_updates":4,"max_seconds":5,"validate_every":1,"patience":5},
                                lr=.01,weight_decay=0.,seed=101,folder=path,device=torch.device("cpu"))
            self.assertTrue((path/"model.pt").is_file())
            n=make_model();n.restore_adaptation(torch.load(path/"model.pt",weights_only=True))
            z=infer_logits(n,x,scale,torch.device("cpu"),3)
            self.assertEqual(z.shape,(9,3));self.assertTrue(np.isfinite(z).all())
            self.assertLessEqual(result["selected_step"],4)
            self.assertEqual(len((path/"trace.csv").read_text().splitlines()),6)

if __name__=="__main__": unittest.main()
