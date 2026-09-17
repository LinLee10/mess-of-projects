import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from recover_artifact import recover,safe_name
class RecoveryControls(unittest.TestCase):
 def make(self,root,corrupt=False,extra=False):
  archive=root/'input.zip';payload=b'checked evidence'
  with zipfile.ZipFile(archive,'w') as z:
   z.writestr('payload.txt',payload)
   z.writestr('ARTIFACT_SHA256.json',json.dumps({'payload.txt':('0'*64 if corrupt else hashlib.sha256(payload).hexdigest())}))
   if extra:z.writestr('unlisted.txt','extra')
  return archive,hashlib.sha256(archive.read_bytes()).hexdigest()
 def test_valid(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a,h=self.make(p);r=recover(a,h,p/'output');self.assertEqual(r['internal_files_checked'],1)
 def test_archive_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a,h=self.make(p)
   with self.assertRaises(ValueError):recover(a,'0'*64,p/'out')
 def test_internal_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a,h=self.make(p,corrupt=True)
   with self.assertRaises(ValueError):recover(a,h,p/'out')
   self.assertFalse((p/'out').exists())
 def test_extra_unlisted(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a,h=self.make(p,extra=True)
   with self.assertRaises(ValueError):recover(a,h,p/'out')
 def test_preserve_existing(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a,h=self.make(p);(p/'out').mkdir()
   with self.assertRaises(FileExistsError):recover(a,h,p/'out')
 def test_unsafe(self):
  for name in ['../escape','/absolute','safe/../../escape','C:\\file','a\\b']:
   self.assertFalse(safe_name(name))
if __name__=='__main__':unittest.main()
