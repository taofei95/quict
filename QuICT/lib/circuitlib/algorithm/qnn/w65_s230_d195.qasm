OPENQASM 2.0;
include "qelib1.inc";
qreg q[65];
creg c[65];
x q[0];
x q[1];
x q[2];
x q[4];
x q[8];
x q[10];
x q[11];
x q[12];
x q[18];
x q[19];
x q[22];
x q[23];
x q[25];
x q[26];
x q[28];
x q[29];
x q[30];
x q[32];
x q[35];
x q[37];
x q[38];
x q[39];
x q[42];
x q[43];
x q[47];
x q[48];
x q[49];
x q[53];
x q[54];
x q[55];
x q[57];
x q[58];
x q[59];
x q[60];
x q[61];
x q[0];
h q[0];
rxx(0.5675429701805115) q[0], q[64];
rxx(0.16196751594543457) q[1], q[64];
rxx(0.7996793985366821) q[2], q[64];
rxx(0.5787332653999329) q[3], q[64];
rxx(0.6384875774383545) q[4], q[64];
rxx(0.39078766107559204) q[5], q[64];
rxx(0.1259250044822693) q[6], q[64];
rxx(0.36508649587631226) q[7], q[64];
rxx(0.47459876537323) q[8], q[64];
rxx(0.7655030488967896) q[9], q[64];
rxx(0.5838963985443115) q[10], q[64];
rxx(0.3265867233276367) q[11], q[64];
rxx(0.03681766986846924) q[12], q[64];
rxx(0.9630906581878662) q[13], q[64];
rxx(0.1352674961090088) q[14], q[64];
rxx(0.5038707256317139) q[15], q[64];
rxx(0.3167232871055603) q[16], q[64];
rxx(0.5627713799476624) q[17], q[64];
rxx(0.8921358585357666) q[18], q[64];
rxx(0.4414832592010498) q[19], q[64];
rxx(0.8974209427833557) q[20], q[64];
rxx(0.5492953658103943) q[21], q[64];
rxx(0.798789381980896) q[22], q[64];
rxx(0.3382560610771179) q[23], q[64];
rxx(0.21594035625457764) q[24], q[64];
rxx(0.490314781665802) q[25], q[64];
rxx(0.8174586296081543) q[26], q[64];
rxx(0.9459140300750732) q[27], q[64];
rxx(0.3254428505897522) q[28], q[64];
rxx(0.47157204151153564) q[29], q[64];
rxx(0.8888623714447021) q[30], q[64];
rxx(0.35501575469970703) q[31], q[64];
rxx(0.6267908811569214) q[32], q[64];
rxx(0.8348315358161926) q[33], q[64];
rxx(0.5586375594139099) q[34], q[64];
rxx(0.6541231870651245) q[35], q[64];
rxx(0.013698041439056396) q[36], q[64];
rxx(0.9206788539886475) q[37], q[64];
rxx(0.22913038730621338) q[38], q[64];
rxx(0.447989821434021) q[39], q[64];
rxx(0.7327590584754944) q[40], q[64];
rxx(0.8408071398735046) q[41], q[64];
rxx(0.4637964367866516) q[42], q[64];
rxx(0.11558175086975098) q[43], q[64];
rxx(0.9497372508049011) q[44], q[64];
rxx(0.6244377493858337) q[45], q[64];
rxx(0.8326751589775085) q[46], q[64];
rxx(0.610519528388977) q[47], q[64];
rxx(0.7109932899475098) q[48], q[64];
rxx(0.8464277982711792) q[49], q[64];
rxx(0.8258278965950012) q[50], q[64];
rxx(0.043952882289886475) q[51], q[64];
rxx(0.43505215644836426) q[52], q[64];
rxx(0.3410256505012512) q[53], q[64];
rxx(0.8066750764846802) q[54], q[64];
rxx(0.9214088916778564) q[55], q[64];
rxx(0.11132407188415527) q[56], q[64];
rxx(0.35365021228790283) q[57], q[64];
rxx(0.24801868200302124) q[58], q[64];
rxx(0.597584068775177) q[59], q[64];
rxx(0.27211517095565796) q[60], q[64];
rxx(0.10773235559463501) q[61], q[64];
rxx(0.4349348545074463) q[62], q[64];
rxx(0.12464302778244019) q[63], q[64];
ryy(0.9494783878326416) q[0], q[64];
ryy(0.8809847831726074) q[1], q[64];
ryy(0.4995155930519104) q[2], q[64];
ryy(0.9719541072845459) q[3], q[64];
ryy(0.9863967895507812) q[4], q[64];
ryy(0.5109595060348511) q[5], q[64];
ryy(0.04354500770568848) q[6], q[64];
ryy(0.5914377570152283) q[7], q[64];
ryy(0.015867650508880615) q[8], q[64];
ryy(0.6584134697914124) q[9], q[64];
ryy(0.5133702754974365) q[10], q[64];
ryy(0.7644743919372559) q[11], q[64];
ryy(0.15535563230514526) q[12], q[64];
ryy(0.12452554702758789) q[13], q[64];
ryy(0.8429673314094543) q[14], q[64];
ryy(0.6407967805862427) q[15], q[64];
ryy(0.9928736090660095) q[16], q[64];
ryy(0.8574380278587341) q[17], q[64];
ryy(0.12155628204345703) q[18], q[64];
ryy(0.6214789748191833) q[19], q[64];
ryy(0.5077778697013855) q[20], q[64];
ryy(0.777585506439209) q[21], q[64];
ryy(0.24651432037353516) q[22], q[64];
ryy(0.325767457485199) q[23], q[64];
ryy(0.9922662973403931) q[24], q[64];
ryy(0.731785237789154) q[25], q[64];
ryy(0.3274924159049988) q[26], q[64];
ryy(0.9202960133552551) q[27], q[64];
ryy(0.9452011585235596) q[28], q[64];
ryy(0.033200979232788086) q[29], q[64];
ryy(0.41988474130630493) q[30], q[64];
ryy(0.9432764053344727) q[31], q[64];
ryy(0.16046059131622314) q[32], q[64];
ryy(0.5778379440307617) q[33], q[64];
ryy(0.27734827995300293) q[34], q[64];
ryy(0.21927040815353394) q[35], q[64];
ryy(0.7211565971374512) q[36], q[64];
ryy(0.8777119517326355) q[37], q[64];
ryy(0.9493132829666138) q[38], q[64];
ryy(0.8187349438667297) q[39], q[64];
ryy(0.7428144216537476) q[40], q[64];
ryy(0.3393747806549072) q[41], q[64];
ryy(0.28466659784317017) q[42], q[64];
ryy(0.36812037229537964) q[43], q[64];
ryy(0.7234614491462708) q[44], q[64];
ryy(0.1643337607383728) q[45], q[64];
ryy(0.06686931848526001) q[46], q[64];
ryy(0.8500967621803284) q[47], q[64];
ryy(0.05929833650588989) q[48], q[64];
ryy(0.16563338041305542) q[49], q[64];
ryy(0.08707112073898315) q[50], q[64];
ryy(0.765804648399353) q[51], q[64];
ryy(0.7770833969116211) q[52], q[64];
ryy(0.5379712581634521) q[53], q[64];
ryy(0.19927841424942017) q[54], q[64];
ryy(0.08962297439575195) q[55], q[64];
ryy(0.1704379916191101) q[56], q[64];
ryy(0.7074006199836731) q[57], q[64];
ryy(0.7073338031768799) q[58], q[64];
ryy(0.9465793967247009) q[59], q[64];
ryy(0.9631711840629578) q[60], q[64];
ryy(0.48220276832580566) q[61], q[64];
ryy(0.027954399585723877) q[62], q[64];
ryy(0.5135043263435364) q[63], q[64];
rzz(0.3897774815559387) q[0], q[64];
rzz(0.9574934840202332) q[1], q[64];
rzz(0.8723358511924744) q[2], q[64];
rzz(0.08453375101089478) q[3], q[64];
rzz(0.055872201919555664) q[4], q[64];
rzz(0.23339545726776123) q[5], q[64];
rzz(0.8041656017303467) q[6], q[64];
rzz(0.28705310821533203) q[7], q[64];
rzz(0.5939203500747681) q[8], q[64];
rzz(0.1891452670097351) q[9], q[64];
rzz(0.5771201848983765) q[10], q[64];
rzz(0.21098530292510986) q[11], q[64];
rzz(0.03632992506027222) q[12], q[64];
rzz(0.09534960985183716) q[13], q[64];
rzz(0.56548672914505) q[14], q[64];
rzz(0.12064403295516968) q[15], q[64];
rzz(0.9343457221984863) q[16], q[64];
rzz(0.4356383681297302) q[17], q[64];
rzz(0.2217646837234497) q[18], q[64];
rzz(0.22140884399414062) q[19], q[64];
rzz(0.30666667222976685) q[20], q[64];
rzz(0.9208862781524658) q[21], q[64];
rzz(0.11662054061889648) q[22], q[64];
rzz(0.8108461499214172) q[23], q[64];
rzz(0.8716581463813782) q[24], q[64];
rzz(0.19208306074142456) q[25], q[64];
rzz(0.7491675019264221) q[26], q[64];
rzz(0.4667539596557617) q[27], q[64];
rzz(0.8752580285072327) q[28], q[64];
rzz(0.4944043755531311) q[29], q[64];
rzz(0.860964298248291) q[30], q[64];
rzz(0.878885805606842) q[31], q[64];
rzz(0.6902244687080383) q[32], q[64];
rzz(0.4189174175262451) q[33], q[64];
rzz(0.7643901109695435) q[34], q[64];
rzz(0.8981419801712036) q[35], q[64];
rzz(0.28460752964019775) q[36], q[64];
rzz(0.7238480448722839) q[37], q[64];
rzz(0.6589311361312866) q[38], q[64];
rzz(0.008886575698852539) q[39], q[64];
rzz(0.3990379571914673) q[40], q[64];
rzz(0.7764672636985779) q[41], q[64];
rzz(0.4077361822128296) q[42], q[64];
rzz(0.1513385772705078) q[43], q[64];
rzz(0.48276299238204956) q[44], q[64];
rzz(0.6824463605880737) q[45], q[64];
rzz(0.7647875547409058) q[46], q[64];
rzz(0.8826179504394531) q[47], q[64];
rzz(0.8995922207832336) q[48], q[64];
rzz(0.6871850490570068) q[49], q[64];
rzz(0.7833867073059082) q[50], q[64];
rzz(0.8357961177825928) q[51], q[64];
rzz(0.3718928098678589) q[52], q[64];
rzz(0.3931726813316345) q[53], q[64];
rzz(0.9329379200935364) q[54], q[64];
rzz(0.7015954256057739) q[55], q[64];
rzz(0.8847154974937439) q[56], q[64];
rzz(0.515932559967041) q[57], q[64];
rzz(0.24122154712677002) q[58], q[64];
rzz(0.9044509530067444) q[59], q[64];
rzz(0.5877119302749634) q[60], q[64];
rzz(0.8784613013267517) q[61], q[64];
rzz(0.9532213807106018) q[62], q[64];
rzz(0.7749772071838379) q[63], q[64];
h q[0];
