OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
sdg q[2];
ry(5.653776028384558) q[5];
u1(6.024971981211341) q[2];
rx(0.0998306620708302) q[1];
ryy(1.6779225654001886) q[1], q[6];
s q[2];
swap q[1], q[3];
id q[2];
u1(4.529224614865605) q[0];
tdg q[2];
rzz(1.6374274888758475) q[4], q[3];
cy q[4], q[0];
ryy(1.7608959291137503) q[8], q[3];
u2(3.901484425483219, 0.030691161018755308) q[4];
p(4.4880052360623095) q[8];
id q[7];
u3(3.634034515721458, 0.4003894978973889, 3.3739091496194997) q[5];
u1(1.3013170576462278) q[4];
tdg q[8];
cu3(0.3327486758517399, 1.850576966911371, 1.2020462946366617) q[9], q[6];
rxx(4.092216943283436) q[8], q[9];
rxx(5.742082875651186) q[8], q[4];
rz(1.8199210639051158) q[5];
id q[0];
p(3.827395322487656) q[3];
u1(0.6338446316415512) q[2];
x q[4];
ryy(5.645673442331898) q[7], q[5];
h q[3];
x q[9];
u1(5.795761698023378) q[0];
s q[2];
u1(3.9065691334549495) q[1];
s q[1];
u3(5.66890514286696, 1.0903500961435635, 3.7300145518174004) q[5];
x q[6];
t q[4];
p(3.1740271929098216) q[8];
s q[1];
id q[0];
u3(1.433612734354902, 6.121785702188071, 4.577277651075176) q[7];
ry(1.7966102142675675) q[9];
tdg q[4];
rz(6.019285289630835) q[4];
tdg q[7];
rxx(2.1195317300601357) q[0], q[1];
rz(1.3418153840407554) q[5];
p(4.269757135250951) q[8];
u3(3.83799516630964, 3.693581095988174, 2.3537313268834854) q[3];
ry(0.33614368393533883) q[4];
cu3(4.313713277807703, 5.761165847234566, 2.9226253408997223) q[2], q[8];
u3(0.14127964942016166, 3.2687286316008204, 4.845946067291253) q[7];
ryy(5.4873857147874405) q[8], q[9];
u3(4.364715365472087, 2.5288201464095135, 0.24390395751490174) q[3];
u1(5.912491182287162) q[2];
rx(5.517128836902499) q[7];
x q[0];
rx(3.885829478418101) q[0];
tdg q[9];
u2(5.735890781014129, 0.24586713183729098) q[5];
h q[3];
rx(2.3328020917071575) q[9];
u2(2.2429899097513615, 2.6958831846385043) q[6];
u1(3.7778621986321617) q[8];
cu3(0.16552943428325156, 0.39500362374446224, 0.05139613714376578) q[5], q[6];
u3(0.9666384249744429, 4.676188966409015, 0.9766159962308439) q[7];
h q[6];
ry(3.600877844645714) q[4];
u1(5.464038016726961) q[2];
u2(4.2585731762206285, 3.734602284291409) q[8];