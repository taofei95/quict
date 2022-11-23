OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
cx q[18], q[3];
rz(5.28295033046302) q[12];
rz(4.922199809013791) q[1];
rz(5.679660502321457) q[6];
cx q[15], q[0];
rz(2.150290577522512) q[4];
cx q[11], q[8];
rz(4.982331334863612) q[7];
cx q[9], q[14];
rz(0.3501457585383146) q[13];
rz(2.8032684814588116) q[2];
rz(2.80413458798471) q[17];
cx q[10], q[16];
cx q[19], q[5];
rz(0.908874350391257) q[4];
rz(4.883663051996334) q[11];
rz(2.2074342999798335) q[7];
rz(4.11849200341214) q[9];
rz(2.750840760398103) q[6];
cx q[1], q[14];
rz(5.214770797256067) q[17];
rz(1.7652182111021235) q[2];
cx q[15], q[16];
rz(2.627801528139597) q[8];
rz(6.083561783176119) q[5];
rz(6.18743759584942) q[12];
rz(3.5362345608402532) q[18];
rz(1.9116828393434486) q[13];
cx q[19], q[10];
cx q[0], q[3];
rz(2.4537275705988457) q[19];
rz(3.265861373437694) q[18];
rz(0.9755470595030088) q[17];
rz(3.629820632173202) q[9];
rz(2.023885328725547) q[10];
rz(6.025263237473036) q[2];
rz(5.763214196512987) q[1];
rz(5.587357870295998) q[6];
cx q[4], q[14];
rz(1.3958904605323852) q[7];
rz(3.2872730708483626) q[12];
rz(2.0274649793470303) q[0];
rz(3.394851160916141) q[15];
rz(4.72531764357158) q[3];
rz(5.498836084711062) q[11];
cx q[8], q[16];
rz(4.992290872304287) q[13];
rz(6.07844044539502) q[5];
rz(2.3962500904204997) q[3];
rz(5.3951148623151735) q[4];
rz(5.123756989836259) q[1];
rz(1.8916679860225176) q[13];
rz(0.1557869110497916) q[16];
rz(2.296388327425707) q[14];
rz(0.4712379433882921) q[12];
cx q[8], q[10];
rz(4.872159653108158) q[17];
rz(2.5554475931464022) q[19];
rz(1.4400381927351085) q[15];
rz(3.513737455916962) q[18];
rz(5.406460739672081) q[2];
rz(1.9354621288169656) q[9];
rz(5.3210872021706335) q[11];
rz(5.049536535593295) q[0];
rz(3.025483320290858) q[7];
cx q[5], q[6];
cx q[9], q[18];
cx q[5], q[12];
cx q[10], q[8];
cx q[6], q[2];
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
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
rz(1.772391818798507) q[1];
rz(4.679560640650893) q[11];
cx q[3], q[0];
rz(2.9125807563725155) q[7];
cx q[14], q[13];
rz(3.6729293870730175) q[16];
rz(5.004241286958628) q[19];
rz(4.580194892825747) q[4];
cx q[17], q[15];
rz(0.7137207259714596) q[9];
cx q[5], q[12];
cx q[1], q[17];
rz(5.985372686548366) q[3];
rz(0.557047307870753) q[19];
rz(3.404511460667215) q[15];
rz(0.8781113918801076) q[14];
rz(3.9036882244657027) q[11];
rz(4.306259641030713) q[16];
rz(6.223224346438717) q[18];
rz(3.4867462190020944) q[6];
rz(2.9146269309007247) q[8];
cx q[7], q[4];
rz(4.871021512964201) q[10];
rz(1.4869866137216416) q[13];
rz(4.121807868490986) q[2];
rz(3.7245403948738374) q[0];
rz(1.4186285629874968) q[11];
rz(1.502234684329797) q[9];
rz(5.377767473386045) q[7];
rz(3.260335544711972) q[1];
rz(3.104170379146232) q[8];
cx q[2], q[18];
rz(5.42746141858728) q[13];
rz(4.571629379834091) q[3];
rz(3.8387400918182863) q[12];
rz(3.925302995791168) q[4];
rz(1.3462532418132223) q[15];
rz(1.4192500621828827) q[14];
rz(4.558024014517248) q[10];
rz(0.8135676587164287) q[16];
cx q[5], q[6];
cx q[0], q[19];
rz(3.305982511007783) q[17];
rz(3.2828530127196287) q[4];
rz(2.007698774233178) q[5];
cx q[17], q[19];
cx q[1], q[8];
rz(2.0452509197651136) q[11];
cx q[6], q[18];
cx q[3], q[16];