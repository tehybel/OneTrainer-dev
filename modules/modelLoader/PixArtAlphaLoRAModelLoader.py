from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.modelLoader.pixartAlpha.PixArtAlphaEmbeddingLoader import PixArtAlphaEmbeddingLoader
from modules.modelLoader.pixartAlpha.PixArtAlphaLoRALoader import PixArtAlphaLoRALoader
from modules.modelLoader.pixartAlpha.PixArtAlphaModelLoader import PixArtAlphaModelLoader
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class PixArtAlphaLoRAModelLoader(
    BaseModelLoader,
    ModelSpecModelLoaderMixin,
    InternalModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.PIXART_ALPHA:
                return "resources/sd_model_spec/pixart_alpha_1.0-lora.json"
            case ModelType.PIXART_SIGMA:
                return "resources/sd_model_spec/pixart_sigma_1.0-lora.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> PixArtAlphaModel | None:
        base_model_loader = PixArtAlphaModelLoader()
        lora_model_loader = PixArtAlphaLoRALoader()
        embedding_loader = PixArtAlphaEmbeddingLoader()

        model = PixArtAlphaModel(model_type=model_type)
        self._load_internal_data(model, model_names.lora)
        model.model_spec = self._load_default_model_spec(model_type)

        if model_names.base_model is not None:
            base_model_loader.load(model, model_type, model_names, weight_dtypes)
        lora_model_loader.load(model, model_names)
        embedding_loader.load(model, model_names.lora, model_names)

        return model
