OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
cx q[19], q[3];
rz(5.942316261475294) q[6];
rz(3.283215832695364) q[5];
cx q[9], q[4];
rz(4.372479893153962) q[18];
rz(1.7076283604031348) q[0];
cx q[15], q[10];
rz(4.142305529839549) q[2];
rz(3.731048720495873) q[13];
rz(2.1531692600520875) q[14];
rz(3.645516331310189) q[1];
rz(4.234811855372516) q[7];
rz(0.3285560910973981) q[11];
rz(1.9878737622012974) q[17];
rz(5.155833223842407) q[12];
rz(5.524630146365755) q[16];
rz(2.6134842131101577) q[8];
rz(0.2343827396543438) q[13];
rz(5.518564891632553) q[11];
cx q[3], q[18];
rz(2.941215908480343) q[7];
rz(1.736427724509991) q[17];
rz(3.9952011235183122) q[6];
rz(0.5313092689657146) q[5];
rz(2.46058709083979) q[16];
cx q[0], q[12];
rz(0.9449634207616459) q[9];
rz(3.2063950191100563) q[1];
rz(1.0374509527515674) q[8];
cx q[15], q[4];
rz(0.5750495796368292) q[14];
cx q[19], q[10];
rz(1.9287526683037155) q[2];
rz(6.157565891918989) q[0];
rz(0.2824312867448858) q[4];
rz(4.386529028883349) q[8];
rz(6.208868426099609) q[7];
rz(4.894249254754805) q[9];
cx q[13], q[15];
rz(5.759123098210603) q[2];
rz(4.80943120474719) q[17];
rz(5.794036084796236) q[14];
cx q[18], q[5];
rz(1.2836772780800518) q[1];
cx q[16], q[19];
rz(2.2138072511311826) q[10];
rz(0.33780688949285637) q[12];
rz(1.7625617851852242) q[6];
rz(5.903469536523865) q[11];
rz(5.269912011873154) q[3];
rz(3.0033758657600256) q[10];
rz(2.859044646658528) q[5];
rz(1.4437476686216377) q[14];
rz(5.730165035029915) q[3];
cx q[18], q[7];
rz(1.7151058381272353) q[2];
rz(0.9200757556721918) q[4];
rz(2.192149943293969) q[9];
rz(5.78925938098213) q[17];
rz(2.8292212284387555) q[16];
cx q[13], q[19];
rz(4.569700619178331) q[15];
rz(3.285048674044677) q[1];
rz(6.260800297223215) q[6];
rz(4.39690357446009) q[11];
cx q[0], q[12];
rz(0.9278539999068766) q[8];
rz(2.02929240600131) q[8];
cx q[4], q[18];
rz(2.071402351712676) q[17];
rz(5.205155890426483) q[12];
rz(4.601223572836778) q[11];
rz(4.771231974921576) q[3];
rz(2.523571182362596) q[14];
rz(0.7509187598759751) q[19];
rz(5.902137624383936) q[1];
cx q[2], q[5];
cx q[16], q[7];
rz(0.41206633752559035) q[9];
cx q[13], q[10];
rz(5.748340812111579) q[0];
rz(5.0147050922631085) q[15];
rz(0.9030850217295325) q[6];
rz(4.465402999421252) q[2];
rz(0.06364119028343018) q[19];
rz(2.684014130867092) q[12];
cx q[15], q[3];
rz(3.909586178877432) q[14];
rz(4.393860323492355) q[1];
rz(0.15222918588304443) q[10];
rz(3.497543017150337) q[18];
rz(0.6795464474151034) q[11];
rz(0.05390236886384057) q[0];
rz(5.947157898539656) q[16];
cx q[4], q[5];
rz(4.553674678693108) q[13];
rz(5.882290561155812) q[17];
rz(4.536626803581361) q[7];
rz(2.88273225234891) q[8];
rz(6.106610858206019) q[9];
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
rz(4.2770287471138895) q[6];
cx q[10], q[9];
cx q[13], q[8];
rz(3.2386327618974766) q[7];
rz(5.123238355193247) q[1];
rz(4.403367365050154) q[5];
rz(3.440228891408981) q[4];
rz(1.593079180439588) q[15];
cx q[17], q[18];
rz(2.038244704571688) q[12];
rz(0.3899301101422275) q[14];
rz(1.818950528544673) q[2];
rz(1.2258055260571412) q[0];
cx q[6], q[11];
rz(3.629201142853312) q[16];
rz(2.7400837292354523) q[3];
rz(0.46182302747332316) q[19];
rz(0.8389421276975045) q[14];
rz(5.155066758765217) q[5];
rz(0.2594646866249895) q[16];
rz(5.91447606655223) q[19];
cx q[0], q[8];
cx q[17], q[4];
rz(2.1287066849125984) q[6];
rz(1.618905784003213) q[1];
rz(5.228719514423062) q[13];
rz(4.7001168066635195) q[18];
rz(2.9122488934283823) q[9];
rz(3.770737162172835) q[10];
rz(3.698863521284871) q[7];
cx q[2], q[15];
rz(4.270817640866082) q[11];
rz(2.358596463267874) q[3];
rz(1.4319576595959325) q[12];
rz(2.7460361579611066) q[8];
rz(6.278169892251561) q[18];
rz(0.27809529495320023) q[7];
rz(6.256476638054649) q[0];
rz(1.0414850592856826) q[10];
rz(6.230097325486327) q[11];
cx q[2], q[19];
rz(2.6304562381942027) q[12];
rz(5.352997312355192) q[17];
rz(5.231664940490732) q[3];
cx q[4], q[14];
rz(5.2244594792810135) q[15];
rz(4.515668131627059) q[16];
rz(5.840854606200514) q[1];
rz(6.137825226329618) q[13];
rz(4.0248414070420715) q[6];
rz(4.532224866053999) q[9];
rz(4.309030063143715) q[5];
rz(1.147487384645123) q[13];
rz(6.224094550457943) q[8];
rz(2.9966106175006337) q[2];
cx q[3], q[12];
rz(5.334903602124458) q[9];
rz(3.9119903163052894) q[16];
rz(0.14454272833713616) q[5];
rz(3.2565950800835726) q[17];
rz(4.738606521998931) q[10];
cx q[4], q[1];
rz(0.2240284725849526) q[7];
rz(6.033385248210747) q[14];
cx q[19], q[18];
rz(4.03577238437337) q[11];
cx q[6], q[15];
rz(1.5341470707860407) q[0];
rz(1.5818539814008352) q[12];
rz(5.874070985143257) q[5];
rz(1.491281221426912) q[10];
cx q[1], q[3];
rz(2.684407374443777) q[15];
cx q[7], q[0];
rz(2.0364859196284435) q[17];
rz(1.3174716094968277) q[13];
rz(0.7268862491550502) q[9];
rz(1.755572395154436) q[6];
rz(1.6520665492914899) q[18];
rz(2.169370985421192) q[4];
