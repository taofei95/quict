OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cy q[0], q[4];
ry(0.1640290144253425) q[1];
tdg q[2];
cu3(3.46767248117927, 2.349387220524077, 4.348888030027799) q[2], q[4];
u2(3.3974355848716784, 4.924487711153334) q[2];
rzz(0.9457131452055249) q[0], q[3];
x q[4];
t q[3];
h q[4];
cz q[4], q[1];
tdg q[3];
sdg q[3];
h q[4];
cx q[4], q[1];
p(5.947288788128438) q[3];
t q[4];
rx(5.097697050180557) q[4];
u1(3.8921972308503023) q[4];
id q[3];
rz(1.3865065864448674) q[2];
h q[1];
ch q[1], q[3];
u3(3.197499511658526, 3.968684120821256, 5.282078469651085) q[3];
ryy(2.191842276521736) q[1], q[3];
rx(0.7667944234623174) q[0];
swap q[2], q[3];
u1(2.7722567953256383) q[3];
p(0.6572593964916617) q[0];
rx(0.1865929340739194) q[3];
sdg q[4];
ryy(3.0638050160937174) q[3], q[1];
cu1(1.8437830254957481) q[3], q[2];
rzz(1.1498415935409383) q[0], q[3];
crz(2.155031206764667) q[3], q[4];
p(3.6765508719353956) q[0];
tdg q[1];
p(5.778010987140865) q[4];
u1(1.8383492512516655) q[4];
s q[4];
tdg q[4];
t q[1];
sdg q[2];
tdg q[1];
p(5.466060179174107) q[0];
cu1(0.7176521802815985) q[3], q[2];
t q[2];
ch q[4], q[2];
id q[2];
u3(3.0137318509753115, 0.755189102625556, 0.9352708087122784) q[3];
u1(1.075246746613476) q[0];
x q[0];
t q[4];
rz(3.991305371035893) q[4];
p(4.970017904311348) q[2];
rxx(1.670867637615248) q[0], q[3];
x q[3];
rxx(5.14747826124741) q[2], q[0];
u2(2.7222984498440317, 5.002003123319947) q[1];
u1(2.5742822401902266) q[2];
u3(5.12796238255037, 1.7313846645178361, 4.758375516668248) q[4];
s q[0];
tdg q[2];
rz(5.610220834750714) q[2];
rz(6.167068186221858) q[4];
id q[3];
ry(1.289628432264929) q[4];
p(3.6166825340101676) q[4];
x q[3];
u1(1.0962848525785132) q[4];
u3(4.216238214295153, 1.215044758313858, 1.0516322396645053) q[1];
x q[0];
u2(2.7755880345497124, 5.161301082293539) q[4];
ry(5.997773266946904) q[0];
id q[1];
x q[2];