import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
# commented out due to deprecated call to AllChem.GetMorganFingerprintAsBitVect
# from rdkit.Chem import AllChem
import autograd.numpy as np
import os
import shutil

def smiles_to_fps(data, fp_length, fp_radius):
    return stringlist2intarray(np.array([smile_to_fp(s, fp_length, fp_radius) for s in data]))

def treat_smarts(smarts):
    s = ""
    for c in smarts:
        if c == "/":
            s += "_"
        else:
            s += c
    return s

def smile_to_fp(s, fp_length, fp_radius, draw=False):
    # print("Calling smile_to_fp for {}".format(s))
    m = Chem.MolFromSmiles(s)
    # Deprecated, keep for documentation only - Yalun
    # result_old = (AllChem.GetMorganFingerprintAsBitVect(
    #     m, fp_radius, nBits=fp_length)).ToBitString()
    fpg = rdFingerprintGenerator.GetMorganGenerator(fp_radius, fpSize=fp_length)
    # Create the AdditionalOutput object to capture the bit info
    ao = rdFingerprintGenerator.AdditionalOutput()
    # Allocate space for the BitInfoMap, which DrawMorganBit needs
    ao.AllocateBitInfoMap()
    fp = fpg.GetFingerprint(m, additionalOutput=ao)
    fpstring = fp.ToBitString()
    # Extract the bit info dictionary
    bit_info = ao.GetBitInfoMap()
    on_bits = list(fp.GetOnBits())
    if draw:
        # Draw
        img_dir = "images/{}".format(treat_smarts(s))
        if os.path.isdir(img_dir):
            shutil.rmtree(img_dir)
        os.mkdir(img_dir)
        for bit_to_draw in on_bits:
            if bit_to_draw == 0:
                continue
            # print(f"Drawing Morgan bit with index: {bit_to_draw}")
            # Draw the specific bit
            # Pass the molecule, the bit index, and the bit info map
            img = Draw.DrawMorganBit(m, bit_to_draw, bit_info, useSVG=True)
            with open("images/{}/bit_{}.svg".format(
                treat_smarts(s), bit_to_draw), 'w') as f:
                f.write(img)
    return fpstring

def stringlist2intarray(A):
    '''This function will convert from a list of strings "10010101" into in integer numpy array.'''
    return np.array([list(s) for s in A], dtype=int)
