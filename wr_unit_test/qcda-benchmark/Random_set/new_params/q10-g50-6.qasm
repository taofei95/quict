OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
ryy(3.105025838289628) q[2], q[6];
ry(2.1180896888712892) q[1];
cz q[2], q[5];
sdg q[7];
u2(5.118205695472418, 4.0974761774348805) q[2];
rx(5.169343133840928) q[2];
u3(5.461715381314946, 2.09249056140837, 0.5696468027173868) q[0];
p(2.5452354649922007) q[4];
u3(5.790329256207491, 3.834738325322276, 5.44678208057712) q[9];
rz(3.331575365804026) q[4];
p(0.5985295233152227) q[2];
tdg q[3];
cx q[1], q[5];
ch q[9], q[7];
rz(6.252861441794496) q[0];
rxx(1.303113514305054) q[0], q[5];
u2(0.12258233748887153, 6.111432620332459) q[7];
p(3.027702547350976) q[3];
x q[2];
u3(0.520945436436176, 4.2155902845806015, 1.82519756052909) q[1];
t q[6];
u1(0.9921169496312553) q[7];
h q[5];
t q[3];
crz(2.3450375643784303) q[3], q[0];
id q[3];
u3(4.25625834082022, 5.526927549195001, 0.938197105847342) q[1];
x q[0];
sdg q[1];
t q[4];
ch q[1], q[3];
rx(5.6996629585167655) q[2];
ry(2.0128279920077548) q[9];
rz(4.647653614991323) q[8];
h q[0];
s q[6];
s q[7];
x q[5];
rx(2.9007404320610126) q[4];
tdg q[4];
t q[7];
crz(3.293373544884534) q[4], q[5];
x q[3];
tdg q[0];
ch q[0], q[5];
s q[2];
cy q[1], q[6];
u2(0.8884256867914789, 5.985897725141521) q[1];
rzz(0.5515477974098874) q[7], q[9];
s q[6];