OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
cx q[0], q[19];
cx q[16], q[25];
rz(2.405004861916413) q[7];
rz(1.8859133365355962) q[15];
rz(3.175247180956692) q[21];
rz(2.4927702091193074) q[24];
rz(5.880946497387866) q[10];
rz(0.35330026462242303) q[17];
rz(6.093303753086712) q[2];
rz(5.956524715090347) q[13];
rz(2.6486357631497137) q[20];
rz(3.8231340488296093) q[18];
cx q[8], q[23];
rz(5.596007937838157) q[22];
rz(4.362799951535602) q[28];
rz(0.6534849400662648) q[9];
rz(5.466897158566878) q[1];
rz(2.5406618529428227) q[11];
rz(5.234280800251989) q[14];
rz(3.8212175853197095) q[29];
rz(4.103644054250019) q[3];
rz(1.4403882021399617) q[6];
rz(1.5359457848744624) q[12];
cx q[26], q[4];
rz(0.9356803962743933) q[27];
rz(0.780381248600839) q[5];
cx q[1], q[15];
rz(0.09254816223885635) q[9];
rz(0.8661483854662132) q[12];
rz(4.617210965173939) q[2];
rz(2.2221994612320866) q[16];
rz(3.751356445774765) q[5];
rz(3.1124232868479833) q[22];
rz(4.772150444952911) q[3];
cx q[17], q[14];
rz(4.474016132600243) q[7];
rz(1.7861331647727958) q[8];
rz(2.4520351143975434) q[4];
cx q[28], q[24];
cx q[19], q[27];
rz(5.42690742472076) q[23];
cx q[20], q[13];
rz(2.066031940458343) q[29];
rz(0.9496333881428678) q[26];
rz(0.3210438400254724) q[6];
rz(2.529314009786048) q[25];
rz(2.122006232128908) q[11];
rz(0.5428001911304057) q[18];
rz(3.157781439726981) q[10];
rz(1.633917364417918) q[21];
rz(6.127304803246343) q[0];
rz(5.120029904475908) q[18];
rz(1.8577368042968272) q[28];
rz(2.964166420918431) q[26];
rz(4.1918626280319655) q[29];
rz(4.0164539233691015) q[24];
rz(2.7122858331441586) q[14];
rz(1.3699525120336107) q[6];
rz(2.18466134980874) q[8];
rz(2.1718351023339704) q[4];
rz(0.21426057749191219) q[9];
rz(0.9784266506705731) q[5];
rz(2.9291738470291953) q[2];
rz(1.0419137437984707) q[3];
rz(5.134872277911837) q[19];
rz(4.645660026998127) q[13];
rz(4.570007581166457) q[27];
rz(6.246798617755291) q[22];
rz(5.93418304421171) q[23];
cx q[12], q[10];
rz(0.45205333234757955) q[15];
rz(6.1061307691756825) q[0];
rz(4.957039218930722) q[21];
cx q[7], q[11];
rz(5.462138161024355) q[17];
rz(4.7278239359828715) q[16];
cx q[25], q[1];
rz(2.1940009595438132) q[20];
rz(3.302686347126807) q[1];
rz(5.339294855615326) q[24];
rz(0.4175463561132103) q[2];
cx q[12], q[27];
cx q[6], q[13];
rz(5.21418272436349) q[16];
rz(2.5217860826431613) q[22];
rz(5.826021015095886) q[21];
rz(0.22669007301286262) q[20];
rz(2.3646800928781095) q[5];
rz(4.687847770722095) q[18];
cx q[29], q[17];
cx q[4], q[8];
rz(0.4755106438439953) q[10];
rz(3.5656687994887686) q[14];
rz(0.7598771526758998) q[11];
rz(2.6795811527739257) q[25];
rz(2.1042466281127603) q[23];
rz(4.136576036898128) q[15];
rz(1.4744555256287595) q[28];
rz(4.099107550732773) q[0];
rz(3.861899144606292) q[3];
rz(1.2192534579486916) q[9];
rz(3.4812109028651994) q[19];
rz(1.4078349268784385) q[26];
rz(5.858827660329281) q[7];
cx q[3], q[6];
rz(1.830591425117224) q[29];
rz(1.7252864879015264) q[16];
rz(0.15569138653323464) q[20];
cx q[11], q[23];
rz(5.426013583789998) q[25];
rz(5.972649036993994) q[9];
rz(5.83836920738818) q[5];
rz(3.126030582092855) q[14];
rz(2.025423964562392) q[7];
rz(4.033659006592941) q[28];
rz(4.8733568985591145) q[19];
cx q[24], q[13];
rz(6.034823463051377) q[4];
rz(2.5751007334587617) q[8];
rz(1.8073298330893959) q[2];
cx q[1], q[0];
rz(1.7314649246535452) q[27];
rz(1.5587220448785397) q[21];
cx q[12], q[17];
rz(2.92031537341674) q[18];
rz(4.120239461484973) q[15];
cx q[10], q[26];
rz(1.663923550547108) q[22];
cx q[16], q[11];
rz(0.35802837282233513) q[24];
rz(5.060277056274082) q[26];
rz(1.9532654792612691) q[23];
rz(6.2727651381437415) q[4];
rz(2.9739152848096695) q[8];
rz(1.2986886550354273) q[13];
rz(1.0943921231699958) q[27];
rz(2.523961148552675) q[2];
cx q[20], q[29];
rz(3.9406049987208784) q[5];
cx q[0], q[18];
rz(4.619198150629698) q[3];
rz(2.219510797680197) q[15];
rz(5.787603193849473) q[12];
cx q[28], q[19];
rz(0.5507970020600853) q[10];
rz(4.689373936970873) q[6];
rz(4.227952562547463) q[7];
rz(2.8515895125241557) q[25];
rz(3.6841353320211825) q[9];
cx q[14], q[21];
rz(5.7723525894390475) q[17];
cx q[1], q[22];
rz(1.7991434325767484) q[13];
rz(6.029838067210554) q[10];
rz(1.6449058594190606) q[27];
rz(5.446060394920857) q[9];
rz(5.75096466675627) q[0];
rz(5.8986205954628925) q[7];
rz(3.1034118958508916) q[2];
rz(5.602615276053499) q[23];
rz(2.056192353765095) q[19];
rz(5.655234608672199) q[21];
rz(3.5422655826842218) q[8];
rz(0.25785437285825435) q[12];
rz(3.426789016217107) q[15];
rz(1.4585249364795332) q[28];
rz(5.993057594846035) q[14];
rz(0.05647446842467492) q[11];
cx q[25], q[4];
rz(1.6268892895943081) q[5];
cx q[17], q[18];
rz(5.888902024731305) q[6];
rz(4.354392863274713) q[22];
cx q[3], q[20];
rz(6.272133644450159) q[24];
rz(4.926407365251737) q[29];
cx q[16], q[1];
rz(5.75897295167577) q[26];
rz(3.9657266334274706) q[7];
rz(3.587081848326562) q[11];
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
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];
measure q[27] -> c[27];
measure q[28] -> c[28];
measure q[29] -> c[29];