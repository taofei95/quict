OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(0.08389592099547864) q[2];
rz(0.5534749147045774) q[9];
rz(2.5293504940262515) q[0];
rz(3.7252502842007065) q[8];
rz(2.745141595698849) q[6];
rz(2.6494410519124574) q[7];
rz(4.106671746462694) q[1];
rz(2.021603094116671) q[5];
rz(4.5761426649892885) q[4];
rz(6.203416033922218) q[3];
rz(3.7028924914628107) q[3];
rz(0.32461472915910494) q[8];
rz(0.6894441169124803) q[5];
rz(1.22914282661063) q[9];
cx q[7], q[2];
rz(0.8585713501963586) q[0];
rz(2.8254247347108015) q[1];
rz(0.1069521341234422) q[4];
rz(0.15258811655664162) q[6];
rz(3.36518972123229) q[7];
rz(2.866358798992132) q[1];
rz(2.6814882053215383) q[5];
rz(5.809415218283525) q[4];
cx q[8], q[6];
rz(3.221343001963867) q[3];
rz(5.547626773712236) q[2];
rz(0.7888023883071092) q[0];
rz(0.42952247825039486) q[9];
rz(2.4757241373281675) q[9];
rz(3.5628332901316746) q[4];
rz(0.3784652754024943) q[0];
rz(3.2081524903630694) q[6];
rz(4.656328823362913) q[5];
rz(3.8596420555027593) q[3];
rz(2.4525778941382477) q[1];
rz(0.803624471485132) q[8];
rz(3.921864474136358) q[7];
rz(1.956249535612543) q[2];
rz(5.808389871456267) q[0];
rz(1.8238979860412035) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
rz(5.159171240609783) q[5];
rz(4.109667860648283) q[7];
rz(2.8487230772608085) q[1];
rz(1.1291947897286978) q[4];
rz(0.39760951172761533) q[9];
cx q[6], q[8];
rz(0.628223014218953) q[2];
rz(3.1799384831310373) q[8];
cx q[6], q[5];
rz(2.10461130085302) q[3];
rz(2.237887364911703) q[9];
cx q[4], q[7];
rz(4.60162022143796) q[2];
rz(1.4338445131865594) q[1];
rz(5.237787102412638) q[0];
cx q[9], q[4];
rz(3.012764589148292) q[7];
cx q[1], q[0];
rz(0.9297020405438637) q[3];
rz(4.155212535743171) q[5];
rz(5.939799878623151) q[2];
rz(5.019145046340474) q[8];
rz(4.771316952032859) q[6];
rz(5.236569958337801) q[3];
rz(4.58881211319856) q[6];
rz(5.8746956952224885) q[4];
cx q[8], q[5];
rz(4.587824593864328) q[7];
rz(4.43643643624153) q[1];
cx q[9], q[0];