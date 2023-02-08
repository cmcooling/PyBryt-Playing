from annotated_solutions import naive_annots, dac_annots, matpow_dac, matpow_naive
from testing import test_matpow
import pybryt

naive_annots.clear()
test_matpow(matpow_naive)

dac_annots.clear()
test_matpow(matpow_dac)

naive_annots.append(pybryt.ForbidImport("numpy"))
dac_annots.append(pybryt.ForbidImport("numpy"))

naive_ref = pybryt.ReferenceImplementation("naive-matpow", naive_annots)
dac_ref = pybryt.ReferenceImplementation("dac-matpow", dac_annots)

naive_ref.dump()
dac_ref.dump()
