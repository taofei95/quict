OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cx q[2], q[4];
ry(4.2696397506796275) q[4];
ry(1.904221763315345) q[4];
rz(4.572746543070607) q[4];
rx(2.974862796530885) q[2];
cx q[2], q[0];
rz(0.5850526280440723) q[3];
rz(5.304390223790977) q[2];
rx(1.9334696343587976) q[4];
rx(6.191982230925759) q[2];
rx(1.9676646393842288) q[4];
rz(0.8445522804300984) q[2];
rz(4.016785559908826) q[3];
rx(2.7199182849016363) q[1];
cx q[2], q[0];
rz(4.4619921421796915) q[2];
cx q[2], q[1];
rz(4.189263092645447) q[3];
rx(5.728457672808146) q[3];
rz(5.879931279800495) q[2];
ry(0.05427262218338014) q[3];
h q[3];
rz(2.8045179343171402) q[1];
h q[3];
ry(3.40557852889411) q[4];
h q[4];
rz(2.703579862541533) q[1];
rz(1.8658164420919703) q[1];
cx q[1], q[4];
ry(3.3502455318940982) q[1];
h q[1];
cx q[2], q[4];
h q[1];
h q[2];
h q[2];
ry(1.0357601207801967) q[2];
rz(5.593815481586847) q[3];
cx q[0], q[4];
ry(0.4486117163872413) q[4];
h q[1];
ry(1.494212721517195) q[1];
cx q[2], q[3];
h q[4];
h q[4];
rz(0.5701925372458239) q[3];
ry(1.055529019691417) q[3];
h q[3];
rz(0.9504272201069972) q[2];
ry(1.118200617870136) q[0];
cx q[4], q[2];