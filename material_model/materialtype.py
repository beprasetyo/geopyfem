from material_model.linear_elastic import LinearElastic
from material_model.von_mises import VonMises


# =============================================================================
# MATERIAL MODEL REGISTRY
# =============================================================================
#
# XML:
#
#   <Material ... model="LinearElastic">
#
# is resolved here, not in main.py.
#
# Future models can be registered after their constitutive implementations
# exist, for example:
#
#   "MohrCoulomb": MohrCoulomb,
#   "ModifiedCamClay": ModifiedCamClay,
#   "BarcelonaBasicModel": BarcelonaBasicModel,
#
# Main.py and the global assembly routine do not need an if/elif chain for
# each constitutive model.
# =============================================================================

MATERIAL_MODELS = {
    "LinearElastic": LinearElastic,
    "VonMises": VonMises,

    # Future:
    # "MohrCoulomb": MohrCoulomb,
    # "ModifiedCamClay": ModifiedCamClay,
    # "BarcelonaBasicModel": BarcelonaBasicModel,
}


def get_material_model_class(model_name):
    """
    Return the Python class registered for an XML material-model name.
    """
    if model_name not in MATERIAL_MODELS:
        implemented = ", ".join(MATERIAL_MODELS.keys())

        raise NotImplementedError(
            f'Material model "{model_name}" is not implemented. '
            f"Currently implemented: {implemented}. "
            "Future models can be registered in material_model/materialtype.py."
        )

    return MATERIAL_MODELS[model_name]



def validate_material_model_interface(model):
    """Validate the formal GeoPyFEM constitutive plug-in contract."""
    integrate = getattr(model, "integrate", None)

    if not callable(integrate):
        raise TypeError(
            f'Material model "{getattr(model, "model_name", type(model).__name__)}" '
            "does not implement integrate(...). Every nonlinear GeoPyFEM "
            "constitutive model must return a ConstitutiveResponse."
        )

    return model

def build_material_models(material_specs, formulation="plane_strain"):
    """
    Build material-model objects indexed by Gmsh Physical Surface region.

    Parameters
    ----------
    material_specs : list of dict
        Materials returned by problem_reader.py.
    formulation : str
        Analysis formulation, e.g. plane_strain.

    Returns
    -------
    models_by_region : dict
        Example:
            {
                "soil1": LinearElastic(...),
                "soil2": LinearElastic(...),
            }

    specs_by_region : dict
        Original XML material dictionaries, retained for names/printing.
    """
    models_by_region = {}
    specs_by_region = {}

    for material in material_specs:
        region = material["region"]
        name = material["name"]
        model_name = material["model"]
        parameters = material["parameters"]

        if region is None:
            raise ValueError(
                f'Material "{name}" does not define region="..." in XML.'
            )

        if region in models_by_region:
            raise ValueError(
                f'More than one XML material is assigned to region "{region}".'
            )

        model_class = get_material_model_class(model_name)

        model = model_class(
            parameters=parameters,
            formulation=formulation,
        )

        validate_material_model_interface(
            model
        )

        models_by_region[region] = model
        specs_by_region[region] = material

    return models_by_region, specs_by_region
