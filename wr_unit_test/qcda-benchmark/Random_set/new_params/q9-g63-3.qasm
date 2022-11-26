OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
s q[4];
sdg q[1];
ch q[5], q[0];
rz(4.8487001789511295) q[5];
u1(1.7865115852966569) q[5];
p(3.0519118399395975) q[2];
p(4.33716535121537) q[8];
cu3(1.297706988269114, 4.324617597956326, 1.3533227674036) q[7], q[2];
rz(3.1858943292056385) q[8];
u3(1.0180876945655248, 0.48213577613838876, 5.562297244539923) q[2];
rx(1.0393361733613498) q[2];
u1(2.0559801743696777) q[3];
rx(3.6915745421589237) q[3];
cx q[2], q[0];
h q[4];
p(0.6997290286308808) q[7];
id q[6];
x q[5];
u1(4.284091940960314) q[2];
tdg q[0];
u1(3.639711754345854) q[1];
tdg q[1];
tdg q[6];
x q[2];
ry(4.0965168770003935) q[4];
s q[5];
cy q[7], q[4];
rx(4.079054614317663) q[3];
u3(5.055482774277895, 0.2740054351421062, 2.654084491139549) q[2];
id q[1];
s q[5];
rz(0.9563661761913232) q[5];
u1(1.321136664282097) q[2];
cz q[0], q[2];
u1(4.942291247256117) q[8];
u1(3.6363207271284237) q[1];
s q[8];
h q[7];
h q[1];
ch q[0], q[6];
id q[0];
rz(2.843471482128594) q[0];
rx(3.2101633545677224) q[0];
x q[4];
rz(3.8912762401762313) q[7];
x q[8];
p(2.8548847772024404) q[1];
rz(3.318930456029885) q[1];
u3(0.6588336386702963, 5.026496563598909, 0.46470017831662513) q[4];
tdg q[0];
x q[5];
rz(3.23106518263554) q[4];
u2(3.516064193195704, 4.632424279541188) q[3];
x q[8];
ry(4.114419455955923) q[7];
tdg q[3];
u1(2.4307721375089857) q[4];
rzz(5.191328713996666) q[3], q[5];
ryy(3.8508854711706864) q[0], q[2];
tdg q[2];
sdg q[8];
s q[5];
rzz(5.584876165502282) q[2], q[6];