OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
cx q[6], q[10];
cx q[11], q[5];
rz(3.985436309870004) q[3];
rz(5.128264550831911) q[2];
rz(4.348722627955487) q[7];
rz(5.559691122274306) q[9];
rz(1.9916824141052298) q[4];
cx q[0], q[8];
rz(4.950420576609413) q[1];
rz(2.2842329831138) q[10];
rz(0.36607999157086) q[4];
cx q[0], q[11];
cx q[3], q[1];
rz(4.993435413553683) q[2];
rz(0.6713511119906541) q[5];
rz(3.36929440152001) q[9];
rz(0.5500085833811731) q[8];
rz(4.11403971650369) q[6];
rz(1.346544394224244) q[7];
rz(5.095410733271405) q[0];
cx q[1], q[3];
cx q[11], q[4];
rz(4.752213055045824) q[9];
rz(4.500980722956757) q[7];
rz(0.7962894243831886) q[8];
rz(3.6002777528107144) q[6];
rz(0.9580263971266663) q[10];
rz(5.225168035209533) q[2];
rz(4.306900158830037) q[5];
rz(2.8265783999705687) q[9];
rz(0.4415576037393489) q[11];
rz(5.7150011601660395) q[8];
rz(4.090792725250367) q[10];
rz(2.565700530959122) q[5];
rz(5.762193850328393) q[4];
rz(5.062539113620192) q[6];
rz(4.566034962953814) q[0];
rz(5.5318358914571) q[1];
rz(1.5730824535883527) q[7];
rz(2.233758969743136) q[2];
rz(3.863762213784009) q[3];
rz(5.4326384663252085) q[2];
rz(5.222538355323391) q[1];
rz(0.9790819420416799) q[8];
rz(3.4248108738383043) q[0];
cx q[5], q[4];
rz(6.252722807221445) q[11];
rz(3.791053094084502) q[7];
rz(6.237422742877718) q[9];
rz(5.331197161280059) q[3];
rz(1.8179189633479897) q[10];
rz(6.131388126158203) q[6];
rz(0.6797300586053635) q[4];
rz(1.7093597326225185) q[10];
cx q[8], q[5];
cx q[7], q[0];
rz(1.84633506672668) q[6];
rz(5.568159044963498) q[9];
rz(5.659084140200726) q[11];
rz(3.502998687519862) q[2];
rz(1.198508025858311) q[3];
rz(0.01000278800225129) q[1];
rz(0.5490090349329156) q[0];
cx q[2], q[8];
rz(5.047972655228978) q[11];
rz(5.746697312563353) q[4];
rz(2.7606392876400503) q[10];
rz(6.232247901061316) q[5];
rz(4.856648018789864) q[1];
rz(3.178621700874057) q[7];
rz(1.8082550827850943) q[9];
rz(1.7542335270678149) q[6];
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
rz(4.619211410486173) q[3];
rz(2.293379884773623) q[11];
rz(4.674581518591254) q[6];
rz(4.1357331299550655) q[9];
rz(1.795958683381525) q[0];
rz(5.888469564752992) q[7];
rz(0.13279230731412772) q[8];
rz(3.6699461973743674) q[5];
rz(4.940191355277916) q[10];
rz(6.0019708991438385) q[4];
rz(2.915080160827105) q[1];
cx q[3], q[2];
rz(3.118644729103134) q[5];
rz(4.511558390951879) q[6];
rz(5.648348500834004) q[3];
rz(3.7129556131017414) q[7];
rz(2.0271839304949593) q[8];
rz(2.722922977582217) q[11];
rz(5.708228460685463) q[1];
rz(4.955660312640609) q[9];
rz(4.354010365674904) q[4];
rz(3.1503390579133748) q[10];
rz(4.30550640263957) q[2];
rz(5.548394969610121) q[0];
rz(1.3664778854222355) q[11];
rz(0.5739928641944416) q[5];
rz(5.406122124937398) q[0];
cx q[3], q[6];
rz(4.221536195760144) q[10];
rz(4.015235957760974) q[1];
rz(2.2849226601299257) q[2];
cx q[9], q[7];
rz(5.0331594786591065) q[8];
rz(5.798057913637125) q[4];
rz(0.05950346385936313) q[9];
rz(4.838258266563942) q[11];
cx q[3], q[5];
rz(3.65020921474574) q[7];
rz(2.3200861646071895) q[1];
rz(5.79563167917932) q[6];
rz(3.451099870602267) q[8];
rz(5.597987778273529) q[0];
cx q[4], q[2];
rz(3.4901395187262554) q[10];
rz(1.8963795547759912) q[1];
rz(0.9510197487683206) q[6];
cx q[9], q[4];
rz(0.8757256083652818) q[2];
rz(2.8756724879166167) q[5];
rz(4.02052512729977) q[8];
rz(1.5353114535787322) q[7];
rz(4.756537186014938) q[3];
rz(0.8545862789139513) q[11];
rz(4.860023000629135) q[0];
rz(3.304486703192452) q[10];
rz(1.5500392478326754) q[1];
rz(4.271637409239733) q[6];
cx q[2], q[4];
rz(3.832227540552044) q[5];
rz(5.1388537876609455) q[11];