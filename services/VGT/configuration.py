import logging
from os.path import join
from pathlib import Path


SRC_PATH = Path(__file__).parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.absolute()

handlers = [logging.StreamHandler()]
logging.root.handlers = []
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
service_logger = logging.getLogger(__name__)


MODELS_PATH = Path(join(ROOT_PATH, "/app/object_detection/weights"))

DOCLAYNET_TYPE_BY_ID = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List_Item",
    5: "Page_Footer",
    6: "Page_Header",
    7: "Picture",
    8: "Section_Header",
    9: "Table",
    10: "Text",
    11: "Title",
}