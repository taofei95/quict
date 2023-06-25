OPENQASM 2.0;
include "qelib1.inc";
qreg q[55];
creg c[55];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
x q[9];
x q[11];
x q[12];
x q[13];
x q[14];
x q[16];
x q[17];
x q[22];
x q[23];
x q[25];
x q[26];
x q[27];
x q[29];
x q[32];
x q[34];
x q[37];
x q[40];
x q[41];
x q[42];
x q[43];
x q[44];
x q[48];
x q[49];
x q[50];
x q[51];
x q[52];
x q[0];
h q[0];
rzz(0.6747633814811707) q[0], q[54];
rzz(0.7296467423439026) q[1], q[54];
rzz(0.898147702217102) q[2], q[54];
rzz(0.8136014938354492) q[3], q[54];
rzz(0.5833885669708252) q[4], q[54];
rzz(0.017661690711975098) q[5], q[54];
rzz(0.5880217552185059) q[6], q[54];
rzz(0.3263595700263977) q[7], q[54];
rzz(0.605840265750885) q[8], q[54];
rzz(0.20620214939117432) q[9], q[54];
rzz(0.16280120611190796) q[10], q[54];
rzz(0.8694161176681519) q[11], q[54];
rzz(0.09203457832336426) q[12], q[54];
rzz(0.45136022567749023) q[13], q[54];
rzz(0.8179017305374146) q[14], q[54];
rzz(0.7625647187232971) q[15], q[54];
rzz(0.4378247857093811) q[16], q[54];
rzz(0.6699389815330505) q[17], q[54];
rzz(0.1382584571838379) q[18], q[54];
rzz(0.8912584781646729) q[19], q[54];
rzz(0.4711250066757202) q[20], q[54];
rzz(0.7905523777008057) q[21], q[54];
rzz(0.28543633222579956) q[22], q[54];
rzz(0.8329117894172668) q[23], q[54];
rzz(0.3249557614326477) q[24], q[54];
rzz(0.12465733289718628) q[25], q[54];
rzz(0.4880925416946411) q[26], q[54];
rzz(0.22927910089492798) q[27], q[54];
rzz(0.09700995683670044) q[28], q[54];
rzz(0.9829420447349548) q[29], q[54];
rzz(0.23208457231521606) q[30], q[54];
rzz(0.5471217036247253) q[31], q[54];
rzz(0.8148179054260254) q[32], q[54];
rzz(0.1589391827583313) q[33], q[54];
rzz(0.018549740314483643) q[34], q[54];
rzz(0.968109130859375) q[35], q[54];
rzz(0.08742916584014893) q[36], q[54];
rzz(0.6634377837181091) q[37], q[54];
rzz(0.3521084785461426) q[38], q[54];
rzz(0.6425891518592834) q[39], q[54];
rzz(0.38019371032714844) q[40], q[54];
rzz(0.843043327331543) q[41], q[54];
rzz(0.6861503720283508) q[42], q[54];
rzz(0.8401870131492615) q[43], q[54];
rzz(0.17665326595306396) q[44], q[54];
rzz(0.2656705379486084) q[45], q[54];
rzz(0.9393438696861267) q[46], q[54];
rzz(0.8986387252807617) q[47], q[54];
rzz(0.23229175806045532) q[48], q[54];
rzz(0.7576068639755249) q[49], q[54];
rzz(0.8101391196250916) q[50], q[54];
rzz(0.7319566011428833) q[51], q[54];
rzz(0.7210494875907898) q[52], q[54];
rzz(0.09903699159622192) q[53], q[54];
rzz(0.0867885947227478) q[0], q[54];
rzz(0.8708685040473938) q[1], q[54];
rzz(0.06788194179534912) q[2], q[54];
rzz(0.47784245014190674) q[3], q[54];
rzz(0.2589094042778015) q[4], q[54];
rzz(0.6744585037231445) q[5], q[54];
rzz(0.31150054931640625) q[6], q[54];
rzz(0.5476106405258179) q[7], q[54];
rzz(0.9925518035888672) q[8], q[54];
rzz(0.1215134859085083) q[9], q[54];
rzz(0.7790977954864502) q[10], q[54];
rzz(0.9621645212173462) q[11], q[54];
rzz(0.5332548022270203) q[12], q[54];
rzz(0.1492377519607544) q[13], q[54];
rzz(0.203632652759552) q[14], q[54];
rzz(0.3325676918029785) q[15], q[54];
rzz(0.2958252429962158) q[16], q[54];
rzz(0.23833829164505005) q[17], q[54];
rzz(0.33477890491485596) q[18], q[54];
rzz(0.4205372929573059) q[19], q[54];
rzz(0.8842961192131042) q[20], q[54];
rzz(0.12291359901428223) q[21], q[54];
rzz(0.9567223191261292) q[22], q[54];
rzz(0.7254019379615784) q[23], q[54];
rzz(0.7737043499946594) q[24], q[54];
rzz(0.6989162564277649) q[25], q[54];
rzz(0.8718290328979492) q[26], q[54];
rzz(0.2015211582183838) q[27], q[54];
rzz(0.16547363996505737) q[28], q[54];
rzz(0.5666092038154602) q[29], q[54];
rzz(0.0507044792175293) q[30], q[54];
rzz(0.9486668109893799) q[31], q[54];
rzz(0.5990208983421326) q[32], q[54];
rzz(0.7201195955276489) q[33], q[54];
rzz(0.9110816121101379) q[34], q[54];
rzz(0.32014185190200806) q[35], q[54];
rzz(0.6977066397666931) q[36], q[54];
rzz(0.6389807462692261) q[37], q[54];
rzz(0.6463126540184021) q[38], q[54];
rzz(0.1813017725944519) q[39], q[54];
rzz(0.4035297632217407) q[40], q[54];
rzz(0.3054932951927185) q[41], q[54];
rzz(0.741603434085846) q[42], q[54];
rzz(0.4217798113822937) q[43], q[54];
rzz(0.5029974579811096) q[44], q[54];
rzz(0.08937418460845947) q[45], q[54];
rzz(0.2979220151901245) q[46], q[54];
rzz(0.2056465744972229) q[47], q[54];
rzz(0.10908818244934082) q[48], q[54];
rzz(0.005373775959014893) q[49], q[54];
rzz(0.6155182719230652) q[50], q[54];
rzz(0.500808835029602) q[51], q[54];
rzz(0.5713551044464111) q[52], q[54];
rzz(0.8248121738433838) q[53], q[54];
h q[0];
