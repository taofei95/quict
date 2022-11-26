OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
ry(0.25361259659872126) q[4];
h q[5];
h q[6];
id q[6];
p(3.5393420757951652) q[9];
ch q[6], q[4];
rx(5.075887702016607) q[9];
cu1(5.214007546428911) q[2], q[9];
id q[6];
t q[2];
crz(1.3082458397064414) q[6], q[4];
p(2.529774218803598) q[0];
rx(0.8460807013507056) q[6];
p(0.20693620968605705) q[6];
cu3(6.039045850680976, 0.9043195493446333, 4.34104599856385) q[1], q[5];
x q[6];
swap q[4], q[1];
u3(4.096023067856541, 5.9944072297229285, 1.9331981096059974) q[2];
s q[7];
id q[7];
sdg q[6];
sdg q[7];
u1(6.021215860116489) q[6];
cy q[3], q[7];
t q[2];
p(1.8089681812298706) q[8];
tdg q[4];
rzz(2.9462371401575234) q[9], q[0];
x q[8];
rxx(3.915854250952046) q[3], q[0];
x q[5];
ry(0.8534116858316074) q[6];
u1(2.967694052396843) q[6];
t q[7];
u3(1.1924274386734655, 5.641348508880083, 6.006390918560412) q[3];
u3(2.8765216020246287, 1.6296720895497323, 3.47732162285957) q[7];
h q[1];
rz(5.974239551847663) q[3];
u2(4.013902758504072, 2.4741488961088383) q[0];
rz(2.58636473946067) q[9];
u3(4.046194860932091, 5.43399021686997, 2.5007667771297766) q[8];
p(6.129053745941932) q[5];
u3(2.822660937071054, 3.2665216729084925, 1.66996400982904) q[8];
u2(4.868668096157841, 3.263284934005433) q[2];
u1(3.459359494160799) q[6];
s q[3];
cu1(2.8215961470806854) q[3], q[2];
tdg q[1];
ryy(5.992519564713288) q[8], q[6];
x q[0];