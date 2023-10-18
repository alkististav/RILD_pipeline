import numpy as np
import nibabel as nib
import scipy as sp


##### Remove class and refactor
class DistanceTransforms():  ######## 3.2!!!!!!!!!

    """
    Class containing different distance transforms for 3D nifti images
    """

    def __init__(self, binNibIn):

        self.nib = binNibIn
        self.affine = binNibIn.affine
        self.header = binNibIn.header

    def euclidian(self):

        """
        :return Euclidian distance transform from binary 3D volume
        """

        neg = np.where(self.nib.get_fdata() == 0, 1, 0)
        dt = sp.ndimage.morphology.distance_transform_edt(neg)

        eucNib = nib.Nifti1Image(dt, affine=self.affine, header=self.header)

        return dt

    def dropoffSDT(self):

        """
        :return 1-(1/1+x) signed distance transform, nan values after 12 voxels
        """

        neg = np.where(self.nib.get_fdata() == 0, 1, 0)

        # eucledian dt
        eucdist = sp.ndimage.morphology.distance_transform_edt(self.nib.get_fdata())
        oppdist = sp.ndimage.morphology.distance_transform_edt(neg)

        # calculate inversely decreasing transform
        dt1 = 1 - (1 / (1 + eucdist))
        dt2 = -(1 - (1 / (1 + oppdist)))
        dt = dt1 + dt2

        dt[eucdist > 12] = float("nan")
        dt[oppdist > 12] = float("nan")

        nibDropoff = nib.Nifti1Image(dt, affine=self.affine, header=self.header)

        return nibDropoff

    def minMaxDropoffSDT(self):

        """
        :return: 1-(1/1+x) signed distance transform, edge values after 12 voxels
        """

        neg = np.where(self.nib.get_fdata() == 0, 1, 0)  # error here

        # eucledian dt
        eucdist = sp.ndimage.morphology.distance_transform_edt(self.nib.get_fdata())
        oppdist = sp.ndimage.morphology.distance_transform_edt(neg)

        # calculate inversely decreasing transform
        dt1 = 1 - (1 / (1 + eucdist))
        dt2 = -(1 - (1 / (1 + oppdist)))
        dt = dt1 + dt2

        if np.amax(eucdist) > 12:

            dt[eucdist > 12] = max(dt[eucdist == 12])
            dt[oppdist > 12] = max(dt[oppdist == 12])

        else:

            dt[oppdist > 12] = max(dt[oppdist == 12])

        nibDropoff = nib.Nifti1Image(dt, affine=self.affine, header=self.header)

        return nibDropoff

