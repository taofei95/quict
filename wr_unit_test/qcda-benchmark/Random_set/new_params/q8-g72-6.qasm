OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
ryy(3.3771913625166112) q[6], q[7];
cx q[2], q[3];
u3(2.908369528048658, 1.2340387301266669, 2.7480531589035766) q[5];
cu1(3.042295475027347) q[4], q[3];
id q[0];
rx(1.1065996356823438) q[7];
ryy(2.1623606979629755) q[7], q[5];
u2(4.688240322316859, 0.5021097540572573) q[6];
u2(0.5623199625411257, 5.31861053332415) q[4];
id q[2];
rz(2.4643038376830555) q[6];
t q[0];
s q[2];
rzz(0.026748034228060838) q[5], q[0];
x q[5];
t q[2];
rz(4.736802070894154) q[0];
id q[4];
h q[7];
h q[7];
p(1.719348604205619) q[0];
tdg q[5];
rz(2.4923714249807447) q[2];
cu3(3.807604551617596, 6.229937736472872, 2.1393273325896995) q[1], q[2];
rxx(4.426644597921286) q[0], q[5];
id q[4];
u1(4.606258299113455) q[3];
u2(3.535482924826176, 4.46005006071922) q[0];
t q[2];
h q[0];
x q[3];
id q[7];
u1(0.5501863870621952) q[4];
id q[4];
rx(4.612761498090121) q[7];
t q[5];
ry(5.724548814293143) q[5];
t q[3];
u2(5.138566537016179, 2.1943498160943475) q[4];
s q[4];
tdg q[1];
t q[6];
rx(0.060503307021660666) q[5];
s q[7];
h q[0];
t q[0];
p(0.10611222437129791) q[6];
ch q[7], q[1];
id q[6];
h q[7];
t q[6];
t q[6];
cz q[4], q[3];
p(4.620943383843369) q[6];
u2(5.497847528032453, 3.8547941286539156) q[6];
u2(1.8946668683074244, 1.89133777268889) q[2];
s q[6];
cy q[5], q[4];
p(5.731300259213934) q[1];
t q[2];
h q[4];
p(0.9251864553703222) q[0];
x q[0];
u3(4.881778161669255, 0.5310682967785676, 4.297749921798851) q[7];
cu3(1.7794956589757516, 4.434241941245577, 1.2338660827614982) q[7], q[5];
rx(1.0802699819955144) q[2];
sdg q[0];
x q[7];
p(5.080899237750084) q[1];
cz q[0], q[7];
s q[5];
h q[7];