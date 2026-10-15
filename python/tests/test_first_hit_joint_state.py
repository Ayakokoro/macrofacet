from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from first_hit_se import FORMAT, VERSION, EVENT, SE_KERNEL
from first_hit_se.batch import losses_many
from first_hit_se.data import read_json
from first_hit_se.gradient_head import cache_profiles
from first_hit_se.joint_state import (epoch_batches,draw_batch,next_trace,cache_distance_terms,
                                     cached_loss,prepare,train_run)
from first_hit_se.model import FirstHitNetwork,ModelConfig
from renewal.collect import digest,write_json


class JointStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_data(self):
        torch.manual_seed(21)
        model=FirstHitNetwork(ModelConfig(hidden=16,width=8,mixture_width=8,components=8,maximum_step=.125))
        p={'id':'toy','split':'train','horizon':1.,
           'features':torch.tensor([[.4,-.4,-1.2,-.7,0.]],dtype=torch.float64),
           'distance':torch.tensor([.1,.3,.6,1.],dtype=torch.float64),
           'hit':torch.tensor([True,True,True,False]),'initial_sign':torch.tensor([1,-1,1,-1]),
           'speed':torch.tensor([.4,.8,1.2,math.nan],dtype=torch.float64)}
        return model,[{**p,'id':str(i)} for i in range(3)]

    def test_sampling_pairs_batches_and_does_not_shift_adjacent_seeds(self):
        _,ps=self.setup_data();settings={'profiles_per_batch':2,'samples_per_profile':3}
        def trace(seed,epoch):
            value='00'*32;seen=[]
            for selection in epoch_batches(ps,settings,seed,epoch):
                value=next_trace(value,selection);seen.extend(i for i,_ in selection)
            self.assertEqual(sorted(seen),[0,1,2])
            return value
        self.assertEqual(trace(12,1),trace(12,1))
        self.assertNotEqual(trace(12,2),trace(13,1))

    def test_frozen_cache_matches_full_joint_loss_and_head_gradients(self):
        source,ps=self.setup_data();source=source.double()
        contexts=cache_profiles(source,ps)
        distances=cache_distance_terms(source,ps,contexts,'cpu')
        model=deepcopy(source)
        selection=[(2,torch.tensor([3,2,0])),(0,torch.tensor([1,3]))]
        direct=sum(r['complete_joint'].sum() for r in losses_many(model,draw_batch(ps,selection)))/5
        direct.backward()
        expected={k:v.grad.clone() for k,v in model.mixture_head.named_parameters()}
        self.assertTrue(all(any(v.grad is not None and v.grad.abs().max()>0 for v in module.parameters())
                            for module in (model.initial_encoder,model.dynamics,model.hazard_head)))
        model.zero_grad(set_to_none=True)
        loss,_,hits,count=cached_loss(model,contexts,distances,selection)
        loss.backward()
        self.assertEqual((hits,count),(3,5))
        torch.testing.assert_close(loss,direct,atol=1e-12,rtol=1e-12)
        for k,v in model.mixture_head.named_parameters():
            torch.testing.assert_close(v.grad,expected[k],atol=1e-12,rtol=1e-12)

    def test_training_modes_share_samples_and_joint_resumes_exactly(self):
        source,ps=self.setup_data()
        header={'format':FORMAT,'version':VERSION,'event':EVENT,'kernel':SE_KERNEL}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'source.pt';data=root/'dataset.pt'
            torch.save({**header,'model_config':source.configuration(),'model_state':source.state_dict()},checkpoint)
            val={**ps[0],'split':'validation','id':'val'}
            torch.save({**header,'profile_mode':'cubic','profiles':ps+[val]},data)
            train_context=cache_profiles(source,ps);val_context=cache_profiles(source,[val])
            torch.save({'identity':{'dataset_sha256':digest(data),'checkpoint_sha256':digest(checkpoint)},
                        'train':train_context},root/'train_context.pt')
            torch.save({'identity':{'dataset_sha256':digest(data),'scene_dataset_sha256':digest(data),
                        'checkpoint_sha256':digest(checkpoint)},'splits':{'validation':val_context}},root/'eval_context.pt')
            spec={'output_directory':str(root/'results'),'model_directory':str(root/'models'),'dataset':str(data),
                  'evaluation_dataset':str(data),'scene_dataset':str(data),'initialize_from':str(checkpoint),
                  'training_contexts':str(root/'train_context.pt'),'evaluation_contexts':str(root/'eval_context.pt'),
                  'device':'cpu','cpu_threads':1,'seeds':[12],'modes':['head_only','joint'],
                  'training':{'epochs':3,'profiles_per_batch':2,'samples_per_profile':4,'learning_rate':.001,
                  'minimum_learning_rate':.0001,'weight_decay':1e-5,'gradient_clip':1.,'validation_every':1,
                  'cuda_graph_solver':False}}
            config=root/'config.json';write_json(config,spec)
            experiment=prepare(config)
            experiment['runs'].append({'name':'joint_resume','mode':'joint','seed':12})
            write_json(root/'results'/'experiment.json',experiment)
            head=train_run(config,'head_only_s12');joint=train_run(config,'joint_s12')
            self.assertEqual(head['sample_trace'],joint['sample_trace'])
            self.assertEqual(head['hits_seen'],joint['hits_seen'])
            self.assertEqual(head['optimizer_updates'],6)
            self.assertEqual(head['parameter_max_changes']['dynamics'],0)
            self.assertTrue(all(v>0 for v in joint['parameter_max_changes'].values()))
            def interrupt(path,value):
                if Path(path).name=='history.json':raise RuntimeError('interrupt checkpoint')
                write_json(path,value)
            with mock.patch('first_hit_se.joint_state.write_json',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupt checkpoint'):
                    train_run(config,'joint_resume')
            resumed=train_run(config,'joint_resume')
            self.assertEqual(resumed['sample_trace'],joint['sample_trace'])
            a=torch.load(root/'models/joint_s12/last.pt',weights_only=True)
            b=torch.load(root/'models/joint_resume/last.pt',weights_only=True)
            for name,value in a['model_state'].items():
                torch.testing.assert_close(value,b['model_state'][name],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
