"""Run openSMILE with ASCII configuration paths in a Unicode workspace."""

from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

import opensmile

__version__ = "0.1.0"


@contextmanager
def egemaps_lld():
    """Yield a 25-feature extractor; keep its temporary configs alive during use.

    openSMILE 2.6 encodes option paths as ASCII and resolves junctions.
    Copy only its bundled configs, not audio or pretrained model weights.
    Use process_signal() with decoded audio to avoid non-ASCII input paths.
    """
    source = Path(opensmile.__file__).resolve().parent / "core" / "config"
    if str(source).isascii():
        yield opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                              feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
        return
    with tempfile.TemporaryDirectory(prefix="emotion-opensmile-") as temporary:
        if not temporary.isascii():
            raise RuntimeError("Set TMP to an ASCII-only directory before running openSMILE.")
        target = Path(temporary) / "config"
        shutil.copytree(source, target)

        class AsciiSmile(opensmile.Smile):
            @property
            def default_config_root(self):
                return str(target)

        yield AsciiSmile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                         feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
