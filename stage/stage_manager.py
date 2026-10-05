from copy import deepcopy


class StageManager:
    """
    Manage cumulative construction-stage state for GeoPyFEM.

    Current stage controls
    ----------------------
    1. Boundary-load activation/deactivation by load name.
    2. Gravity activation/deactivation and gravity-vector changes.
    3. Domain-region activation/deactivation by Gmsh Physical Surface name.
    4. Sequential phase history.
    5. Optional displacement-reference reset after a converged stage.

    Region activity
    ---------------
    All domain regions are ACTIVE when StageManager is created.

    Therefore a first construction stage can define initially absent soil or
    structural regions by deactivating them before any mechanical solve:

        <DeactivateRegions>
          <Region name="timbunan_bawah"/>
          <Region name="timbunan_atas"/>
        </DeactivateRegions>

    A later stage may activate one of those regions.

    This keeps construction-region identity tied directly to the Gmsh
    Physical Surface names already used by GeoPyFEM.

    Notes
    -----
    Region activity is cumulative just like boundary-load activity. If a
    region is activated and no later stage changes it, it remains active.
    """

    def __init__(
        self,
        load_definitions,
        region_names,
        initial_gravity=None,
    ):
        self._loads = list(
            load_definitions
            or []
        )

        self._loads_by_name = {}

        for load in self._loads:
            name = load.get(
                "name"
            )

            if not name:
                continue

            if name in self._loads_by_name:
                raise ValueError(
                    f'Duplicate load name "{name}" in <Loads>.'
                )

            self._loads_by_name[
                name
            ] = load

        self._active_load_names = set()

        self._region_names = list(
            dict.fromkeys(
                region_names
                or []
            )
        )

        self._region_name_set = set(
            self._region_names
        )

        # All mesh regions exist/are active before the first stage. Regions
        # that should not initially exist are deactivated in Stage 0.
        self._active_region_names = set(
            self._region_names
        )

        initial_gravity = (
            initial_gravity
            or {}
        )

        self._gravity = {
            "enabled": bool(
                initial_gravity.get(
                    "enabled",
                    False,
                )
            ),
            "gx": float(
                initial_gravity.get(
                    "gx",
                    0.0,
                )
                or 0.0
            ),
            "gy": float(
                initial_gravity.get(
                    "gy",
                    -9.81,
                )
                if initial_gravity.get(
                    "gy"
                ) is not None
                else -9.81
            ),
        }

        self.current_stage = None

    def apply_stage(
        self,
        stage,
    ):
        """
        Apply one stage definition to the cumulative construction state.

        Returns
        -------
        context : dict
            Contains both the cumulative ACTIVE region set and the regions
            whose state changed specifically in this stage.
        """
        activate_regions = list(
            stage.get(
                "activate_regions",
                [],
            )
        )

        deactivate_regions = list(
            stage.get(
                "deactivate_regions",
                [],
            )
        )

        overlap = (
            set(activate_regions)
            & set(deactivate_regions)
        )

        if overlap:
            raise ValueError(
                "A construction stage cannot activate and deactivate the "
                "same region at once. Conflicting regions: "
                f"{sorted(overlap)}"
            )

        for name in activate_regions:
            self._validate_region_name(
                name
            )

        for name in deactivate_regions:
            self._validate_region_name(
                name
            )

        # Record only ACTUAL changes. Repeating "activate" for a region that
        # is already active is harmless but does not create a second birth.
        newly_deactivated = [
            name
            for name in deactivate_regions
            if name in self._active_region_names
        ]

        newly_activated = [
            name
            for name in activate_regions
            if name not in self._active_region_names
        ]

        for name in newly_deactivated:
            self._active_region_names.discard(
                name
            )

        for name in newly_activated:
            self._active_region_names.add(
                name
            )

        for name in stage.get(
            "activate_loads",
            [],
        ):
            self._validate_load_name(
                name
            )

            self._active_load_names.add(
                name
            )

        for name in stage.get(
            "deactivate_loads",
            [],
        ):
            self._validate_load_name(
                name
            )

            self._active_load_names.discard(
                name
            )

        stage_gravity = stage.get(
            "gravity"
        )

        if stage_gravity is not None:
            if stage_gravity.get(
                "enabled"
            ) is not None:
                self._gravity[
                    "enabled"
                ] = bool(
                    stage_gravity[
                        "enabled"
                    ]
                )

            if stage_gravity.get(
                "gx"
            ) is not None:
                self._gravity[
                    "gx"
                ] = float(
                    stage_gravity[
                        "gx"
                    ]
                )

            if stage_gravity.get(
                "gy"
            ) is not None:
                self._gravity[
                    "gy"
                ] = float(
                    stage_gravity[
                        "gy"
                    ]
                )

        self.current_stage = deepcopy(
            stage
        )

        return {
            "stage_id": stage.get(
                "id"
            ),
            "stage_name": stage.get(
                "name"
            ),
            "procedure": stage.get(
                "procedure"
            ),
            "load_steps": stage.get(
                "load_steps"
            ),
            "active_load_names": (
                self.active_load_names
            ),
            "active_loads": (
                self.active_loads
            ),
            "gravity": (
                self.gravity
            ),
            "active_region_names": (
                self.active_region_names
            ),
            "activated_regions": list(
                newly_activated
            ),
            "deactivated_regions": list(
                newly_deactivated
            ),
            "k0": deepcopy(
                stage.get(
                    "k0"
                )
            ),
            "reset_displacements": bool(
                stage.get(
                    "reset_displacements",
                    False,
                )
            ),
        }

    @property
    def active_load_names(
        self,
    ):
        return [
            load.get(
                "name"
            )
            for load in self._loads
            if load.get(
                "name"
            ) in self._active_load_names
        ]

    @property
    def active_loads(
        self,
    ):
        return [
            load
            for load in self._loads
            if load.get(
                "name"
            ) in self._active_load_names
        ]

    @property
    def active_region_names(
        self,
    ):
        return [
            name
            for name in self._region_names
            if name in self._active_region_names
        ]

    @property
    def gravity(
        self,
    ):
        return dict(
            self._gravity
        )

    def _validate_load_name(
        self,
        name,
    ):
        if name not in self._loads_by_name:
            available = sorted(
                self._loads_by_name.keys()
            )

            raise ValueError(
                f'Stage refers to unknown load "{name}". '
                f"Available named loads: {available}"
            )

    def _validate_region_name(
        self,
        name,
    ):
        if name not in self._region_name_set:
            available = sorted(
                self._region_name_set
            )

            raise ValueError(
                f'Stage refers to unknown construction region "{name}". '
                f"Available mesh regions: {available}"
            )
