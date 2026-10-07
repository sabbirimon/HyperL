import hashlib,stat,tempfile,unittest,zipfile
from pathlib import Path
from install import install

class InstallTest(unittest.TestCase):
    def archive(self,directory,entries):
        file=directory/'release.zip'
        with zipfile.ZipFile(file,'w') as z:
            for name,data in entries:z.writestr(name,data)
        return file,hashlib.sha256(file.read_bytes()).hexdigest()
    def test_verified_install_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);file,sha=self.archive(d,[('hyperl-0.1/bin/hyperl','launcher'),('hyperl-0.1/bin/hyperl.bat','launcher')])
            installed=install(file,d/'prefix',sha);self.assertTrue((installed/'bin/hyperl').is_file())
            with self.assertRaises(ValueError):install(file,d/'prefix',sha)
    def test_traversal_symlink_hash_and_case_collision(self):
        for entries in [[('../escape','x')],[('hyperl-0.1/../../escape','x')],[('hyperl-0.1/A','x'),('hyperl-0.1/a','x')]]:
            with tempfile.TemporaryDirectory() as directory:
                d=Path(directory);file,sha=self.archive(d,entries)
                with self.assertRaises(ValueError):install(file,d/'prefix',sha)
                self.assertFalse((d/'prefix').exists())
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);entry=zipfile.ZipInfo('hyperl-0.1/link');entry.external_attr=(stat.S_IFLNK|0o777)<<16
            file,sha=self.archive(d,[(entry,'../../outside')])
            with self.assertRaises(ValueError):install(file,d/'prefix',sha)
            with self.assertRaises(ValueError):install(file,d/'prefix','0'*64)

if __name__=='__main__':unittest.main()
