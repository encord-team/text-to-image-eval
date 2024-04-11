from sources.models.types.open_clip import OpenCLIPModel


class LocalCLIPModel(OpenCLIPModel):
    def __init__(
        self,
        title: str,
        device: str | None = None,
        *,
        title_in_source: str,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(title, device, title_in_source=title_in_source, cache_dir=cache_dir, **kwargs)

    def _setup(self, **kwargs) -> None:
        self.pretrained = (self._cache_dir / "checkpoint.pt").as_posix()
        super()._setup(**kwargs)
