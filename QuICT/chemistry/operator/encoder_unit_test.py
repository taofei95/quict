from fermion_operator import FermionOperator
from encoder import JordanWigner
from encoder import Parity
from encoder import BravyiKitaev

anni_2 = FermionOperator('2')
crea_2 = FermionOperator('2^')
anni_5 = FermionOperator('5')
crea_5 = FermionOperator('5^')

#JW
print("\nJordan Wigner\n")
print(JordanWigner.encoder(anni_2, 7).parse())
print(JordanWigner.encoder(crea_2, 7).parse())
print(JordanWigner.encoder(anni_5, 7).parse())
print(JordanWigner.encoder(crea_5, 7).parse())
print(JordanWigner.encoder(crea_2*anni_2, 7).parse())
print(JordanWigner.encoder(crea_5*anni_5, 7).parse())

print(JordanWigner.encoder(FermionOperator('99'), 100).parse())

#Parity
print("\nParity\n")
print(Parity.encoder(anni_2, 10).parse())
print(Parity.encoder(crea_2, 10).parse())
print(Parity.encoder(anni_5, 10).parse())
print(Parity.encoder(crea_5, 10).parse())
print(Parity.encoder(crea_2*anni_2, 7).parse())
print(Parity.encoder(crea_5*anni_5, 7).parse())

print(JordanWigner.encoder(crea_5+anni_5, 10).parse())
print(Parity.encoder(crea_5+anni_5, 10).parse())


#BK
print("\nBravyi Kitaev\n")
print(BravyiKitaev.encoder(crea_5+anni_5, 10).parse())
print(BravyiKitaev.encoder(FermionOperator('17')+FermionOperator('17^'), 100).parse())
print(BravyiKitaev.encoder(FermionOperator('50')+FermionOperator('50^'), 100).parse())
print(BravyiKitaev.encoder(FermionOperator('73')+FermionOperator('73^'), 100).parse())

print(BravyiKitaev.encoder(anni_2, 10).parse())
print(BravyiKitaev.encoder(crea_2, 10).parse())
print(BravyiKitaev.encoder(anni_5, 10).parse())
print(BravyiKitaev.encoder(crea_5, 10).parse())
print(BravyiKitaev.encoder(crea_2*anni_2, 10).parse())
print(BravyiKitaev.encoder(crea_5*anni_5, 10).parse())