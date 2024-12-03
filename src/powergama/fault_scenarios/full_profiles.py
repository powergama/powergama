import copy

import numpy as np


class FullProfiles:
    """
    FullProfiles is used to store the full profiles in a data structure, to avoid repeatedly reading
    in from file between fault situations.
    """

    def readProfileDataAndStore(
        self, griddata, filename, full_timerange, storagevalue_filling=None, storagevalue_time=None, timedelta=1.0
    ):
        self.stored_profiles = griddata._readProfileFromFile(filename, full_timerange)
        self.timedelta = timedelta
        if storagevalue_filling is not None:
            self.stored_storagevalue_time = griddata._readProfileFromFile(storagevalue_time, full_timerange)
            self.storagevalue_filling = griddata._readStoragevaluesFromFile(storagevalue_filling)

    def setProfileData(self, timerange, griddata):
        """timerange is here relative to full_timerange.
        If full_timerange does not start at zero this can give wrong results
        """
        profiles = self.stored_profiles.loc[timerange]
        profiles.index = range(len(timerange))
        griddata.profiles = profiles

        if "stored_storagevalue_time" in self.__dict__:
            storagevalue_time = self.stored_storagevalue_time.loc[timerange]
            storagevalue_time.index = range(len(timerange))
            griddata.storagevalue_time = storagevalue_time

        griddata.timerange = timerange
        griddata.timeDelta = self.timedelta

    def get_multiplier_demand(self):
        if "_multiplier_demand" in self.__dict__.keys():
            return self._multiplier_demand
        multiplier_demand = {}
        for demand_ref in self.stored_profiles.columns:
            if demand_ref == "index":
                print("Skipping index")
            elif demand_ref[:4] == "load":
                sum_demand = np.sum(self.stored_profiles[demand_ref])
                multiplier_demand[demand_ref] = sum_demand
        self._multiplier_demand = multiplier_demand
        return self._multiplier_demand


def get_gridmodel_failure(full_profiles, gridmodel_base, timerange, failed_generators, failed_branches):
    """Get a version of the gridmodel that takes into account the fault situation.

    timerange : relative to full model

    """

    gridmodel = copy.deepcopy(gridmodel_base)
    full_profiles.setProfileData(timerange, gridmodel)

    # Set failed generators to zero
    for gg in failed_generators:
        # if gg is an int, set that generator to zero
        if type(gg) == int:
            gridmodel.generator.at[gg, "pmax"] = 0
        elif type(gg) == tuple:
            # otherwise we've specified both the generator and the fraction that goes offline
            g_idx, g_down = gg
            gridmodel.generator.at[g_idx, "pmax"] *= 1 - g_down
        else:
            raise ValueError
    # Set failed branches to zero
    # for bb in failed_branches:
    #     gridmodel.branch.at[bb, 'capacity'] = 0
    # Remove failed branches
    to_keep = [ii for ii in range(gridmodel_base.numBranches()) if ii not in failed_branches]
    gridmodel.branch = copy.deepcopy(gridmodel_base.branch.iloc[to_keep])
    gridmodel.branch = gridmodel.branch.reset_index()

    return gridmodel
