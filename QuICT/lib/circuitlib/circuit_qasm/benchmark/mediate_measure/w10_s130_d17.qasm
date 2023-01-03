OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(1.6068129176073036) q[1];
cx q[7], q[9];
rz(2.2646595411173465) q[3];
cx q[0], q[6];
rz(4.598267025190371) q[8];
cx q[5], q[2];
rz(0.647575239104647) q[4];
rz(5.48052698501205) q[5];
cx q[6], q[9];
cx q[3], q[0];
rz(1.0487829201821626) q[2];
rz(4.146874457384841) q[1];
cx q[7], q[8];
rz(5.5919956724684905) q[4];
cx q[4], q[9];
rz(0.44578382352530366) q[2];
rz(0.02877058761574192) q[5];
cx q[0], q[8];
rz(5.544604912018639) q[1];
cx q[6], q[3];
rz(4.8020144231144215) q[7];
cx q[3], q[8];
rz(5.540096003018127) q[6];
rz(2.8662073700787922) q[5];
rz(2.447067070902657) q[9];
rz(3.686103170421897) q[1];
rz(0.43584764372563173) q[4];
cx q[0], q[2];
rz(5.77253833406771) q[7];
cx q[0], q[5];
rz(0.04388968351552556) q[4];
cx q[2], q[9];
rz(1.468097280462091) q[3];
rz(1.4954044145901118) q[1];
rz(4.588952669707206) q[6];
rz(5.0403852976078705) q[7];
rz(3.982363397758977) q[8];
rz(2.6855080466461603) q[1];
rz(2.9033396485068423) q[0];
rz(5.031443819853317) q[2];
rz(4.485723891166345) q[6];
cx q[9], q[5];
rz(0.86650261676104) q[4];
rz(0.7143993631723792) q[3];
rz(1.6820195360015262) q[7];
rz(2.8519246871987387) q[8];
rz(2.129828532260002) q[0];
cx q[2], q[1];
rz(4.737373443790223) q[8];
rz(3.7123064616105705) q[6];
rz(1.6706524857879628) q[9];
rz(6.021802480637041) q[5];
rz(1.7514014802278277) q[7];
rz(1.5787729897964) q[4];
rz(5.122809429271013) q[3];
rz(2.1083222242893482) q[1];
rz(1.185077516933856) q[5];
rz(0.6798662842178307) q[4];
rz(5.1323717670682845) q[7];
rz(5.711017502021117) q[2];
rz(3.3885127223040423) q[8];
rz(1.1555249635504121) q[9];
rz(4.487586789978722) q[0];
rz(2.8692550878342353) q[6];
rz(5.488297573378145) q[3];
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
rz(3.159402532142777) q[5];
rz(3.6890429524345443) q[2];
rz(4.315111952879821) q[7];
rz(4.335853286431497) q[1];
rz(0.7436641135238541) q[3];
cx q[0], q[4];
rz(2.5632486114739423) q[6];
cx q[9], q[8];
rz(4.898730488699063) q[0];
rz(0.40681798850373296) q[7];
cx q[2], q[1];
rz(5.63478634901568) q[8];
rz(3.028084702885513) q[3];
rz(3.220891075806132) q[5];
cx q[6], q[9];
rz(5.072887751447032) q[4];
cx q[8], q[3];
rz(1.3959587961069362) q[6];
rz(3.114656219457544) q[4];
cx q[5], q[7];
rz(1.516096992434927) q[9];
rz(2.0480993030125365) q[1];
cx q[0], q[2];
rz(3.8434919353116284) q[1];
cx q[4], q[2];
rz(3.2501394237391743) q[5];
rz(6.091032524940821) q[0];
cx q[7], q[3];
rz(1.5967135719953338) q[9];
rz(1.8019532870879795) q[8];
rz(2.0859560280053198) q[6];
rz(1.021124789708586) q[7];
cx q[5], q[9];
rz(1.3993752886632271) q[6];
rz(5.476113726283442) q[2];
rz(0.805365355772345) q[4];
rz(2.178406125359823) q[3];
rz(5.683560036761586) q[1];
rz(5.980112626279244) q[8];
rz(4.1161734250231286) q[0];
cx q[8], q[4];
cx q[5], q[1];
rz(0.5115117100276996) q[3];
rz(3.6937679606986626) q[7];
cx q[2], q[9];
rz(4.815095550684948) q[6];
rz(1.349802230317045) q[0];
rz(2.9705304541074105) q[6];
cx q[5], q[1];
cx q[2], q[9];
rz(5.576479980384877) q[4];
rz(4.586173692857183) q[7];
rz(3.118419747567498) q[0];
cx q[8], q[3];
rz(5.101984828819059) q[8];