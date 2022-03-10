"""OpenNMT module."""

from my_opennmt.version import __version__, _check_tf_version

_check_tf_version()

from my_opennmt.config import convert_to_v2_config
from my_opennmt.config import load_config
from my_opennmt.config import load_model

from my_opennmt.constants import END_OF_SENTENCE_ID
from my_opennmt.constants import END_OF_SENTENCE_TOKEN
from my_opennmt.constants import PADDING_ID
from my_opennmt.constants import PADDING_TOKEN
from my_opennmt.constants import START_OF_SENTENCE_ID
from my_opennmt.constants import START_OF_SENTENCE_TOKEN
from my_opennmt.constants import UNKNOWN_TOKEN

from my_opennmt.runner import Runner
import my_opennmt.models
