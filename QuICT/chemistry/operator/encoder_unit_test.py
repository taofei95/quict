from fermion_operator import FermionOperator
from encoder import JordanWigner
from encoder import Parity
from encoder import BravyiKitaev

#JW
anni_2 = FermionOperator('2')
crea_2 = FermionOperator('2^')
anni_5 = FermionOperator('5')
crea_5 = FermionOperator('5^')

print(JordanWigner.encoder(anni_2, 7))