"""Negative controls for the independent checker, without production imports."""
import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from audit_scale import check_scores,check_seal

class AuditorControls(unittest.TestCase):
    def fixture(self):
        return np.array([0,0,0,1]),np.array([[.8,.2],[.7,.3],[.1,.9],[.2,.8]]),dict(n=4,correct=3,accuracy=.75,balanced_accuracy=5/6,macro_f1=(.8+2/3)/2,log_loss=-sum(map(np.log,[.8,.7,.1,.8]))/4,confusion_matrix=[[2,1],[0,1]])
    def test_hand_counted_truth(self):check_scores(*self.fixture())
    def test_corrupt_score_is_rejected(self):
        y,p,s=self.fixture();s['balanced_accuracy']=1
        with self.assertRaises(ValueError):check_scores(y,p,s)
    def test_negative_and_fractional_labels_rejected(self):
        y,p,s=self.fixture()
        for labels in [np.array([-1,0,0,1]),np.array([0.,0.,.9,1.])]:
            with self.assertRaises(ValueError):check_scores(labels,p,s)
    def test_modified_sealed_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'x').write_text('before')
            m={'files':{'x':hashlib.sha256(b'before').hexdigest()}}
            m['manifest_sha256']=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            (root/'FROZEN.json').write_text(json.dumps(m));check_seal(root)
            (root/'x').write_text('after')
            with self.assertRaises(ValueError):check_seal(root)

if __name__=='__main__':unittest.main()
