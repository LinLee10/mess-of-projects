"""Second pass regressions, written before changing the submitted implementation.

Expected outcomes encode input and provenance contracts, not favored method scores.
"""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
import torch
from synthlab_next.common import validate_xy, seal, write_json
from synthlab_next.students import classification_metrics, StudentConfig, train_student
from synthlab_next.portfolio import utility_weights, support_mask
from synthlab_next.inference import predict_csv
from synthlab_next.quality import screen_wine_csv
from synthlab_next.analysis import analyze


class ReviewValidationTests(unittest.TestCase):
    def test_fractional_evaluation_labels_are_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            classification_metrics(np.array([0.0, 1.9]), np.array([[.9,.1],[.1,.9]]))

    def test_boolean_evaluation_labels_are_not_silently_reinterpreted(self):
        with self.assertRaises(ValueError):
            classification_metrics(np.array([False, True]), np.array([[.9,.1],[.1,.9]]))

    def test_evaluation_labels_must_be_one_dimensional(self):
        with self.assertRaises(ValueError):
            classification_metrics(np.array([[0], [1]]), np.array([[.9,.1],[.1,.9]]))

    def test_unknown_evaluation_class_is_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics(np.array([-1, 1]), np.array([[.9,.1],[.1,.9]]))

    def test_complex_features_are_not_silently_projected_to_real(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(ValueError):
                validate_xy(np.array([[1+5j],[2+9j]]), np.array([0,1]))

    def test_portfolio_temperature_must_be_finite(self):
        with self.assertRaises(ValueError):
            utility_weights([[.1,.3],[.3,.1]], [5,5], temperature=float('nan'))

    def test_portfolio_prior_must_be_finite(self):
        with self.assertRaises(ValueError):
            utility_weights([[.1,.3],[.3,.1]], [5,5], prior_strength=float('inf'))

    def test_support_threshold_must_be_finite(self):
        with self.assertRaises(ValueError):
            support_mask(np.array([[0.,0.],[1.,1.],[2.,2.]]), np.array([[.2,.2]]), upper_multiplier=float('nan'))

    def test_metrics_match_hand_counted_confusion_matrix(self):
        y=np.array([0,0,0,1]);p=np.array([[.8,.2],[.7,.3],[.1,.9],[.2,.8]])
        scores=classification_metrics(y,p)
        self.assertAlmostEqual(scores['accuracy'],.75)
        self.assertAlmostEqual(scores['balanced_accuracy'],(2/3+1)/2)
        self.assertEqual(scores['confusion_matrix'],[[2,1],[0,1]])

    def test_prediction_csv_duplicate_names_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);torch.set_num_threads(1)
            x=np.array([[0.,1.],[1.,2.],[2.,3.],[3.,4.]])
            model=train_student(x,np.array([0,0,1,1]),StudentConfig(steps=2,width=4,batch_size=4))
            model.save(root/'student')
            write_json(root/'student/schema.json',{'feature_names':['a','b'],'target_names':['x','y']})
            (root/'input.csv').write_text('a,b,b\n1,2,999\n')
            with self.assertRaises(ValueError):
                predict_csv(root/'student',root/'input.csv',root/'output.csv')
            self.assertFalse((root/'output.csv').exists())

    def test_ragged_schema_csv_leaves_no_partial_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write_json(p/'schema.json',{'dataset':'wine','feature_names':['a','b'],'target_names':['x','y']})
            (p/'input.csv').write_text('a,b,class_id\n1,2,0,extra\n')
            with self.assertRaises(ValueError):
                screen_wine_csv(p/'input.csv',p/'schema.json',p/'out')
            self.assertFalse((p/'out').exists())

    def test_analysis_requires_matching_candidate_and_evaluation_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);run=p/'run';run.mkdir()
            (run/'protocol.json').write_text('{}')
            seal(run,['protocol.json'])
            write_json(run/'final_evaluation/scores.json',{'frozen_manifest_sha256':'incorrect','rows':[]})
            seal(run/'final_evaluation',['scores.json'])
            with self.assertRaises(ValueError):
                analyze(run,p/'analysis')
            self.assertFalse((p/'analysis').exists())


if __name__=='__main__':
    unittest.main()
