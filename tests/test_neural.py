import importlib.util
import unittest
import numpy as np
import pandas as pd

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from volatility.neural import prepare, neural_forecast, qlike_objective, positive
    from volatility.neural_experiment import run_neural, align_reference
from volatility.horizons import horizon_frame
from volatility.experiment import loss


@unittest.skipUnless(HAS_TORCH, 'Install requirements-neural.txt for neural tests')
class NeuralIntegrityTests(unittest.TestCase):
    def protocol(self):
        return dict(hidden_width=4,epochs=6,learning_rate=.01,weight_decay=.001,gradient_clip=5.,seeds=[17,43],glm_alpha=.001)

    def frame(self):
        dates=pd.bdate_range('2000-01-03','2003-03-31')
        series=pd.Series(np.exp(np.random.default_rng(8).normal(-9,.3,len(dates))),index=dates)
        return series, horizon_frame(series,5)

    def test_training_scaler_independent_of_prediction_batch(self):
        _,frame=self.frame()
        train,rows=frame.iloc[:600],frame.iloc[600:610]
        first=prepare(train,rows)
        changed=rows.copy()
        changed[['daily','weekly','monthly']]*=100
        second=prepare(train,changed)
        np.testing.assert_array_equal(first[0],second[0])
        self.assertEqual(first[3],second[3])
        self.assertAlmostEqual(first[3]['target_scale'],train.target.mean())

    def test_objective_equals_qlike_up_to_observation_constant(self):
        y=np.array([.4,2.,5.])
        means=np.array([.8,1.,4.])
        actual=float(qlike_objective(torch.tensor(np.log(means)),torch.tensor(y)))
        self.assertAlmostEqual(actual,loss(y,means).mean()+np.log(y).mean()+1)

    def test_fixed_seed_repeatability_and_positive_outputs(self):
        _,frame=self.frame()
        a,info=neural_forecast(frame.iloc[:600],frame.iloc[600:610],self.protocol(),17)
        b,_=neural_forecast(frame.iloc[:600],frame.iloc[600:610],self.protocol(),17)
        np.testing.assert_array_equal(a,b)
        self.assertTrue((a>0).all())
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.isfinite(info['final_training_objective']))

    def test_learns_a_known_positive_conditional_mean(self):
        _,frame=self.frame()
        frame=frame.copy()
        frame['target']=2*frame.daily
        protocol=self.protocol()
        protocol['epochs']=200
        values,info=neural_forecast(frame.iloc[:600],frame.iloc[600:610],protocol,17)
        baseline=np.repeat(frame.iloc[:600].target.mean(),10)
        self.assertLess(loss(frame.iloc[600:610].target,values).mean(),loss(frame.iloc[600:610].target,baseline).mean())
        self.assertLess(info['final_training_objective'],info['initial_training_objective'])

    def test_invalid_outputs_fail_instead_of_clipping(self):
        for values in [[0.],[-1.],[np.nan],[np.inf]]:
            with self.assertRaises(ValueError):
                positive(values)

    def test_full_pipeline_future_invariance_maturity_and_ensemble(self):
        series,frame=self.frame()
        cfg=dict(development_start='2003-01-01',development_end='2003-03-31',holdout_start='2004-01-01')
        a,fits=run_neural(frame,5,cfg,self.protocol(),progress=False)
        changed=series.copy()
        changed.loc['2003-02-03':]*=7
        b,_=run_neural(horizon_frame(changed,5),5,cfg,self.protocol(),progress=False)
        pd.testing.assert_frame_equal(a[a.origin<'2003-02-03'].drop(columns=['observed']).reset_index(drop=True),b[b.origin<'2003-02-03'].drop(columns=['observed']).reset_index(drop=True))
        self.assertTrue((a.training_target_max<=a.fit_origin).all())
        for fit in fits:
            self.assertLessEqual(fit['training_target_max'],fit['fit_origin'])
        pivot=a.pivot(index='origin',columns='model',values='forecast')
        np.testing.assert_allclose(pivot['MLP ensemble'],pivot[['MLP seed 17','MLP seed 43']].mean(axis=1),rtol=1e-14)

    def test_holdout_rejected(self):
        _,frame=self.frame()
        cfg=dict(development_start='2003-01-01',development_end='2003-03-31',holdout_start='2003-02-01')
        with self.assertRaisesRegex(ValueError,'holdout'):
            run_neural(frame,5,cfg,self.protocol(),progress=False)

    def test_reference_alignment_rejects_changed_outcomes(self):
        dates=pd.bdate_range('2000-01-03',periods=3)
        new=pd.DataFrame(dict(origin=dates,target_end=dates,observed=[1.,2.,3.],model='Gamma GLM'))
        ref=pd.concat([new.assign(model='HAR'),new.assign(model='XGBoost Gamma')])
        align_reference(ref,new)
        ref.loc[ref.model=='HAR','observed']*=2
        with self.assertRaises(AssertionError):
            align_reference(ref,new)
