OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
cx q[0], q[1];
rz(5.131453029392076) q[0];
rz(4.597114053673103) q[1];
cx q[0], q[1];
rz(4.968771711498465) q[1];
rz(0.4498462762517525) q[0];
rz(4.717062308442314) q[1];
rz(0.7559524575967457) q[0];
rz(4.772762235451738) q[1];
rz(5.80971325658065) q[0];
rz(0.19294127452224497) q[1];
rz(1.8971758829842582) q[0];
rz(1.8873261813009343) q[0];
rz(2.8168746431489216) q[1];
rz(2.3453972340988214) q[0];
rz(1.123106438749874) q[1];
cx q[0], q[1];
rz(1.1154416847800923) q[0];
rz(1.2319359830552028) q[1];
rz(0.32580811348295946) q[1];
rz(5.174534878699546) q[0];
rz(3.434096890943202) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
rz(5.1552489225824285) q[0];
rz(0.686729941719441) q[0];
rz(1.4092415068446036) q[1];
rz(5.995270913161651) q[0];
rz(2.0097975085534605) q[1];
rz(1.0000976376075663) q[1];
rz(4.12422237902166) q[0];
cx q[0], q[1];
cx q[0], q[1];
cx q[1], q[0];
rz(5.843265472412654) q[1];
rz(4.198442933777463) q[0];
rz(5.137582602193466) q[0];
rz(5.6128545808126145) q[1];
cx q[1], q[0];
rz(5.290877874232059) q[0];
rz(5.961386070151728) q[1];
rz(3.7925098068028436) q[1];
rz(2.711042138049027) q[0];
rz(1.6758103032477996) q[1];