from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import netCDF4
from matplotlib.axes import Axes

from halo_reader.metadata import Metadata
from halo_reader.type_guards import is_none_list
from halo_reader.variable import Variable


@dataclass(slots=True)
class Halo:
    metadata: Metadata
    time: Variable
    range: Variable
    azimuth: Variable
    elevation: Variable
    pitch: Variable
    roll: Variable
    doppler_velocity: Variable
    intensity: Variable
    beta: Variable
    spectral_width: Variable | None = None

    def to_nc(self) -> memoryview:
        nc = netCDF4.Dataset("inmemory.nc", "w", memory=1028)
        self.time.nc_create_dimension(nc)
        self.range.nc_create_dimension(nc)
        for attr_name in self.__dataclass_fields__.keys():
            halo_attr = getattr(self, attr_name)
            if halo_attr is not None:
                halo_attr.nc_write(nc)
        nc_buf = nc.close()
        if isinstance(nc_buf, memoryview):
            return nc_buf
        raise TypeError

    @classmethod
    def merge(cls, halos: list[Halo]) -> Halo | None:
        if len(halos) == 0:
            return None
        if len(halos) == 1:
            return halos[0]
        halo_attrs: dict[str, Any] = {}
        for attr_name in cls.__dataclass_fields__.keys():
            halo_attr_list = [getattr(h, attr_name) for h in halos]
            if Metadata.is_metadata_list(halo_attr_list):
                halo_attrs[attr_name] = Metadata.merge(halo_attr_list)
            elif Variable.is_variable_list(halo_attr_list):
                halo_attrs[attr_name] = Variable.merge(halo_attr_list)
            elif is_none_list(halo_attr_list):
                halo_attrs[attr_name] = None
            else:
                raise TypeError
        return Halo(**halo_attrs)

    def plot(self, title: str | None = None, ax: Axes | None = None) -> Axes:
        if title is None:
            title = "intensity"
        if ax is None:
            _, ax = plt.subplots(1, 1)
        d_bound = 1e-2
        clim = (1 - d_bound, 1 + d_bound)
        ax.pcolor(
            self.time.data, self.range.data, self.intensity.data.T, clim=clim
        )
        ax.set_title(title)
        return ax


@dataclass(slots=True)
class HaloBg:
    time: Variable
    range: Variable
    background: Variable

    def to_nc(self) -> memoryview:
        nc = netCDF4.Dataset("inmemory.nc", "w", memory=1028)
        self.time.nc_create_dimension(nc)
        self.range.nc_create_dimension(nc)
        for attr_name in self.__dataclass_fields__.keys():
            halobg_attr = getattr(self, attr_name)
            if halobg_attr is not None:
                halobg_attr.nc_write(nc)
        nc_buf = nc.close()
        if isinstance(nc_buf, memoryview):
            return nc_buf
        raise TypeError

    @classmethod
    def merge(cls, halobgs: list[HaloBg]) -> HaloBg | None:
        if len(halobgs) == 0:
            return None
        if len(halobgs) == 1:
            return halobgs[0]
        halobg_attrs: dict[str, Any] = {}
        for attr_name in cls.__dataclass_fields__.keys():
            halobg_attr_list = [getattr(h, attr_name) for h in halobgs]
            if Variable.is_variable_list(halobg_attr_list):
                halobg_attrs[attr_name] = Variable.merge(halobg_attr_list)
            elif is_none_list(halobg_attr_list):
                halobg_attrs[attr_name] = None
            else:
                raise TypeError
        return HaloBg(**halobg_attrs)

    @classmethod
    def is_bgfilename(cls, filename: str) -> bool:
        return filename.lower().startswith("background_")

    def plot(self, title: str | None = None, ax: Axes | None = None) -> Axes:
        if title is None:
            title = "background"
        if ax is None:
            _, ax = plt.subplots(1, 1)
        mean = self.background.data.mean()
        std = self.background.data.std()
        clim = (mean - std, mean + std)
        ax.pcolor(
            self.time.data, self.range.data, self.background.data.T, clim=clim
        )
        ax.set_title(title)
        return ax
