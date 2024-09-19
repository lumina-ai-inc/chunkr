import logging
from os.path import join
from pathlib import Path


SRC_PATH = Path(__file__).parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.absolute()

handlers = [logging.StreamHandler()]
logging.root.handlers = []
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
service_logger = logging.getLogger(__name__)

IMAGES_ROOT_PATH = Path(join(ROOT_PATH, "images"))
WORD_GRIDS_PATH = Path(join(ROOT_PATH, "word_grids"))
JSONS_ROOT_PATH = Path(join(ROOT_PATH, "jsons"))
JSON_TEST_FILE_PATH = Path(join(JSONS_ROOT_PATH, "test.json"))
MODELS_PATH = Path(join(ROOT_PATH, "models"))
XMLS_PATH = Path(join(ROOT_PATH, "xmls"))

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
