"""Publication boundaries and schema screening requirements, not model rankings."""
import csv,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from synthlab_next.common import array_hash,write_json
from synthlab_next.generators import Copula
from synthlab_next.inference import load_generator
from synthlab_next.quality import screen_wine_csv

spec=importlib.util.spec_from_file_location('publication',Path(__file__).parents[1]/'scripts/publish_to_github.py')
publication=importlib.util.module_from_spec(spec);spec.loader.exec_module(publication)

class DeliveryContracts(unittest.TestCase):
    def test_publication_rejects_unrelated_or_unwritable_repositories(self):
        good={'full_name':'LinLee10/synthetic-data-lab','fork':True,'parent':{'full_name':'Arhaan2/synthetic-data-lab'},'permissions':{'push':True}}
        publication.validate_destination(good)
        for bad in [{**good,'full_name':'someone/other'},{**good,'parent':{'full_name':'other/repo'}},{**good,'permissions':{'push':False}}]:
            with self.assertRaises(ValueError):publication.validate_destination(bad)

    def test_publication_dry_run_has_no_external_calls(self):
        with patch.object(publication,'run',side_effect=AssertionError('External call')):
            publication.main(['--workdir','/tmp/unused-research-dryrun'])

    def test_publication_never_overwrites_or_copies_secret_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'src';target=Path(tmp)/'target';source.mkdir();(source/'.env').write_text('placeholder')
            with self.assertRaises(ValueError):publication.copy_extension(source,target)
            self.assertFalse(target.exists());(source/'.env').unlink();(source/'README.md').write_text('safe')
            publication.copy_extension(source,target)
            with self.assertRaises(ValueError):publication.copy_extension(source,target)

    def test_portfolio_reload_and_reference_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rng=np.random.default_rng(8);y=np.repeat([0,1],12);x=rng.normal(size=(24,2))+y[:,None]
            Copula(7).fit(x,y).save(root/'component');np.savez(root/'reference.npz',x=x,y=y)
            m={'method':'portfolio','n_features':2,'n_classes':2,'class_prob':[.5,.5],
                'components':{'copula':'component'},'audit':{},'weighted':False,'gated':False,
                'development_losses':[[1,1]],'development_counts':[5,5],
                'reference_x_sha256':array_hash(x),'reference_y_sha256':array_hash(y)}
            write_json(root/'model.json',m)
            a=load_generator(root);b=load_generator(root);labels=np.array([0,1]*6)
            xa,ya=a.sample_labels(labels,seed=5);xb,yb=b.sample_labels(labels,seed=5)
            np.testing.assert_array_equal(xa,xb);np.testing.assert_array_equal(ya,yb)
            x[0,0]+=1;np.savez(root/'reference.npz',x=x,y=y)
            with self.assertRaises(ValueError):load_generator(root)

    def test_portfolio_refuses_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_json(p/'model.json',{'method':'portfolio','n_features':2,'n_classes':2,'class_prob':[.5,.5],
                'audit':{},'components':{'bad':'../outside'}})
            with self.assertRaises(ValueError):load_generator(p)

    def test_schema_screen_preserves_valid_rows_and_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_json(p/'schema.json',{'dataset':'wine','feature_names':['a','b'],'target_names':['x','y']})
            (p/'input.csv').write_text('a,b,class_id\n1,2,0\n-1,3,1\nnan,4,0\n5,6,2\n')
            result=screen_wine_csv(p/'input.csv',p/'schema.json',p/'screen')
            self.assertEqual((result['accepted_rows'],result['rejected_rows']),(1,3))
            self.assertEqual((p/'screen/schema_checked.csv').read_text(),'a,b,class_id\n1,2,0\n')
            self.assertIn('negative:a',(p/'screen/rejected.csv').read_text());self.assertIn('nonfinite:a',(p/'screen/rejected.csv').read_text())
            with self.assertRaises(ValueError):screen_wine_csv(p/'input.csv',p/'schema.json',p/'screen')

    def test_schema_screen_rejects_ambiguous_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_json(p/'schema.json',{'dataset':'wine','feature_names':['a','b'],'target_names':['x','y']})
            (p/'input.csv').write_text('a,a,class_id\n1,2,0\n')
            with self.assertRaises(ValueError):screen_wine_csv(p/'input.csv',p/'schema.json',p/'out')

if __name__=='__main__':unittest.main()
