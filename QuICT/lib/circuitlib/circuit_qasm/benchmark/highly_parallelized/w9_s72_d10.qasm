OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
rz(0.2767230727416405) q[8];
cx q[6], q[3];
rz(4.8638886257588165) q[5];
rz(5.321322205509209) q[4];
rz(2.742624175224636) q[1];
cx q[0], q[7];
rz(2.1545349721684675) q[2];
rz(5.556275914843151) q[0];
rz(5.2185845916155635) q[7];
rz(6.171096324135401) q[3];
rz(5.279616733724311) q[5];
rz(0.03407850659297214) q[8];
rz(5.411238722906189) q[1];
rz(4.015675906143128) q[6];
cx q[2], q[4];
rz(4.7017840990432305) q[6];
rz(2.1634125050817405) q[5];
rz(4.3836574267931026) q[4];
rz(3.101771268068575) q[1];
rz(3.034209987446892) q[3];
rz(2.2456868705172663) q[2];
cx q[8], q[7];
rz(2.607647195919547) q[0];
rz(4.831262359697991) q[0];
rz(4.831252077409) q[1];
rz(5.451430455168032) q[3];
rz(4.228309669079526) q[7];
rz(5.333773851491116) q[5];
rz(3.2860723446779976) q[2];
rz(2.2833850814285017) q[8];
rz(3.3676512771096716) q[6];
rz(2.0950900476751904) q[4];
cx q[5], q[1];
cx q[4], q[3];
rz(2.8111289418225054) q[8];
rz(4.987524365791859) q[6];
rz(3.3823886881904217) q[2];
cx q[7], q[0];
cx q[2], q[8];
rz(1.8727675079356878) q[0];
rz(3.108006842510542) q[4];
rz(5.296650060707265) q[6];
rz(5.086848373315114) q[1];
rz(0.40326538044771426) q[5];
rz(6.014559968004001) q[7];
rz(6.232494482687826) q[3];
rz(0.13576869422601937) q[8];
rz(4.480777788544227) q[2];
rz(4.511922293172623) q[1];
rz(5.611937387098397) q[6];
rz(1.6871856043515727) q[5];
rz(3.2793309580678187) q[4];
cx q[3], q[7];
rz(2.764786231403438) q[0];
rz(2.7656340481265005) q[0];
cx q[3], q[5];
rz(4.134207669868854) q[8];
rz(2.125818147420324) q[7];
rz(3.5480797625355733) q[4];
rz(2.7915517902004567) q[2];
rz(1.7419226394633096) q[6];
rz(2.765525510595251) q[1];
rz(2.905303312528475) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
