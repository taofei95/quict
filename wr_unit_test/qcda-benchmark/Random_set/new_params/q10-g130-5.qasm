OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
x q[5];
x q[3];
ry(4.761606301945629) q[9];
u3(6.107685606041954, 3.2712917540413184, 3.3536238787090644) q[0];
sdg q[5];
x q[4];
s q[3];
ry(6.013519031959841) q[5];
sdg q[6];
x q[0];
id q[0];
rx(1.6520612613684635) q[0];
t q[7];
u3(6.281607591022595, 3.176472628061548, 1.0427792947239642) q[2];
ryy(3.471584067394172) q[4], q[5];
id q[0];
rx(4.2524824013251425) q[7];
u2(5.178232112087025, 3.174085258199472) q[5];
id q[1];
swap q[1], q[4];
s q[4];
ry(2.463044290940505) q[2];
rzz(5.688463295468929) q[7], q[1];
cu1(1.0465016400763465) q[1], q[5];
ryy(3.644764679698533) q[6], q[0];
rz(1.2993970699476476) q[8];
rx(1.2713240515829634) q[9];
cu1(3.3381675665226114) q[2], q[4];
p(6.06842654808061) q[8];
u2(2.0034884765403174, 3.6162142160854627) q[1];
rxx(0.25533789932901546) q[0], q[9];
u1(4.1894518345996525) q[4];
h q[1];
h q[6];
cy q[1], q[8];
s q[0];
rz(4.30893231805398) q[4];
p(5.207157100772256) q[7];
u2(2.9164163383230988, 1.6631486644484186) q[7];
rx(3.9980773490875037) q[0];
ry(6.193088053469166) q[7];
rx(0.5009397345772478) q[3];
h q[3];
s q[1];
rz(5.327927703010597) q[9];
u2(3.4074633144790014, 2.0114985091815814) q[7];
u1(1.5984940838439143) q[5];
rz(2.494942294218667) q[7];
id q[5];
id q[0];
rx(4.4279073921352206) q[0];
id q[4];
ch q[7], q[2];
cy q[8], q[7];
rz(5.147972602715554) q[8];
u2(0.9075744656310218, 1.4207067095255335) q[1];
x q[0];
x q[1];
x q[1];
u2(2.9320216942861563, 4.583038400180364) q[2];
cz q[0], q[6];
u1(4.437288697406535) q[4];
x q[8];
id q[5];
cz q[0], q[8];
rx(2.468340355884966) q[0];
swap q[2], q[1];
id q[4];
s q[9];
id q[5];
x q[9];
h q[4];
p(5.87894752991046) q[4];
cx q[3], q[2];
rxx(3.1338044884669443) q[5], q[9];
x q[0];
x q[2];
u2(1.2716434926070928, 5.90075939965443) q[7];
p(2.5250367577794153) q[4];
sdg q[8];
u1(5.945798814288377) q[4];
tdg q[7];
rz(5.153947334143244) q[9];
cu1(1.831331938299737) q[9], q[3];
u2(1.9280647700168447, 1.3299690585230948) q[1];
h q[8];
sdg q[5];
ry(2.7733052831951284) q[8];
sdg q[3];
h q[3];
ry(4.265982444807987) q[6];
u2(4.255638172788001, 0.6213120822889695) q[9];
u3(4.142860863416737, 6.04262298938146, 5.637757236809453) q[0];
cy q[1], q[9];
x q[2];
p(5.945443203822908) q[9];
rz(3.9473168689754887) q[6];
p(2.678879884259396) q[4];
rx(4.863416523536651) q[5];
rx(3.6421960939307527) q[1];
u1(3.5284275156456655) q[6];
x q[7];
id q[2];
cx q[0], q[2];
rz(5.123860873361298) q[5];
cx q[4], q[8];
x q[3];
s q[6];
u2(1.7991339180746428, 0.7481873540281766) q[9];
crz(0.18581795038968157) q[6], q[8];
x q[6];
cz q[5], q[3];
sdg q[7];
tdg q[1];
rx(2.6492881662465977) q[6];
t q[3];
rz(0.5483762759120061) q[9];
u3(2.8171825914670516, 1.5126509669336572, 0.7536315695479999) q[3];
rxx(1.4054375958705445) q[3], q[5];
x q[6];
x q[4];
id q[5];
cu1(4.975008056760544) q[3], q[8];
s q[9];
rx(4.6037934447147935) q[5];
ry(2.8547431188263395) q[5];
cy q[7], q[4];
id q[8];
x q[2];
rzz(1.530800119841796) q[8], q[7];