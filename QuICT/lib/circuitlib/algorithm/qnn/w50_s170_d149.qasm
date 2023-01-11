OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
creg c[50];
x q[4];
x q[6];
x q[7];
x q[8];
x q[9];
x q[13];
x q[16];
x q[19];
x q[20];
x q[23];
x q[30];
x q[34];
x q[35];
x q[38];
x q[41];
x q[43];
x q[44];
x q[45];
x q[46];
x q[48];
x q[0];
h q[0];
ryy(0.8376249074935913) q[0], q[49];
ryy(0.240439772605896) q[1], q[49];
ryy(0.95255446434021) q[2], q[49];
ryy(0.8680224418640137) q[3], q[49];
ryy(0.6627371907234192) q[4], q[49];
ryy(0.7048316597938538) q[5], q[49];
ryy(0.8252056241035461) q[6], q[49];
ryy(0.14803016185760498) q[7], q[49];
ryy(0.8181815147399902) q[8], q[49];
ryy(0.13558900356292725) q[9], q[49];
ryy(0.3587709069252014) q[10], q[49];
ryy(0.9252572655677795) q[11], q[49];
ryy(0.8085697293281555) q[12], q[49];
ryy(0.9480385780334473) q[13], q[49];
ryy(0.017940938472747803) q[14], q[49];
ryy(0.7872071266174316) q[15], q[49];
ryy(0.5945467948913574) q[16], q[49];
ryy(0.1474320888519287) q[17], q[49];
ryy(0.14421015977859497) q[18], q[49];
ryy(0.3890998959541321) q[19], q[49];
ryy(0.9739609360694885) q[20], q[49];
ryy(0.9206661581993103) q[21], q[49];
ryy(0.8083756566047668) q[22], q[49];
ryy(0.03425782918930054) q[23], q[49];
ryy(0.8301038146018982) q[24], q[49];
ryy(0.7423683404922485) q[25], q[49];
ryy(0.976800262928009) q[26], q[49];
ryy(0.15749621391296387) q[27], q[49];
ryy(0.7218106389045715) q[28], q[49];
ryy(0.1257249116897583) q[29], q[49];
ryy(0.6993609070777893) q[30], q[49];
ryy(0.7008407711982727) q[31], q[49];
ryy(0.9666703939437866) q[32], q[49];
ryy(0.19488590955734253) q[33], q[49];
ryy(0.9786142706871033) q[34], q[49];
ryy(0.35617536306381226) q[35], q[49];
ryy(0.05225914716720581) q[36], q[49];
ryy(0.154974102973938) q[37], q[49];
ryy(0.9537500739097595) q[38], q[49];
ryy(0.7953745722770691) q[39], q[49];
ryy(0.029192626476287842) q[40], q[49];
ryy(0.16838574409484863) q[41], q[49];
ryy(0.3821106553077698) q[42], q[49];
ryy(0.19381296634674072) q[43], q[49];
ryy(0.09260499477386475) q[44], q[49];
ryy(0.15249764919281006) q[45], q[49];
ryy(0.21112895011901855) q[46], q[49];
ryy(0.5644389390945435) q[47], q[49];
ryy(0.4519774317741394) q[48], q[49];
rzz(0.5424450039863586) q[0], q[49];
rzz(0.6239075064659119) q[1], q[49];
rzz(0.002244889736175537) q[2], q[49];
rzz(0.6875205636024475) q[3], q[49];
rzz(0.5845351219177246) q[4], q[49];
rzz(0.6021194458007812) q[5], q[49];
rzz(0.058866143226623535) q[6], q[49];
rzz(0.32680922746658325) q[7], q[49];
rzz(0.260111927986145) q[8], q[49];
rzz(0.42056524753570557) q[9], q[49];
rzz(0.9417581558227539) q[10], q[49];
rzz(0.7413715124130249) q[11], q[49];
rzz(0.31960105895996094) q[12], q[49];
rzz(0.36163681745529175) q[13], q[49];
rzz(0.22549694776535034) q[14], q[49];
rzz(0.8596656918525696) q[15], q[49];
rzz(0.9274905920028687) q[16], q[49];
rzz(0.6302264332771301) q[17], q[49];
rzz(0.8313387632369995) q[18], q[49];
rzz(0.8191962838172913) q[19], q[49];
rzz(0.18958646059036255) q[20], q[49];
rzz(0.7522712349891663) q[21], q[49];
rzz(0.5494461059570312) q[22], q[49];
rzz(0.6632587909698486) q[23], q[49];
rzz(0.30211228132247925) q[24], q[49];
rzz(0.7071264982223511) q[25], q[49];
rzz(0.6341769695281982) q[26], q[49];
rzz(0.8181356191635132) q[27], q[49];
rzz(0.6130490303039551) q[28], q[49];
rzz(0.8771935701370239) q[29], q[49];
rzz(0.7122166752815247) q[30], q[49];
rzz(0.3246503472328186) q[31], q[49];
rzz(0.5449199676513672) q[32], q[49];
rzz(0.6940319538116455) q[33], q[49];
rzz(0.23586642742156982) q[34], q[49];
rzz(0.9893145561218262) q[35], q[49];
rzz(0.30408287048339844) q[36], q[49];
rzz(0.9460655450820923) q[37], q[49];
rzz(0.16340303421020508) q[38], q[49];
rzz(0.32774537801742554) q[39], q[49];
rzz(0.08501434326171875) q[40], q[49];
rzz(0.6586887240409851) q[41], q[49];
rzz(0.3727272152900696) q[42], q[49];
rzz(0.8543720841407776) q[43], q[49];
rzz(0.8331783413887024) q[44], q[49];
rzz(0.730745792388916) q[45], q[49];
rzz(0.2996808886528015) q[46], q[49];
rzz(0.2607213258743286) q[47], q[49];
rzz(0.8517578840255737) q[48], q[49];
rzx(0.6848092675209045) q[0], q[49];
rzx(0.9523942470550537) q[1], q[49];
rzx(0.4639226794242859) q[2], q[49];
rzx(0.7951725721359253) q[3], q[49];
rzx(0.20348644256591797) q[4], q[49];
rzx(0.2744303345680237) q[5], q[49];
rzx(0.8632541298866272) q[6], q[49];
rzx(0.30579471588134766) q[7], q[49];
rzx(0.005400896072387695) q[8], q[49];
rzx(0.4064077138900757) q[9], q[49];
rzx(0.37100422382354736) q[10], q[49];
rzx(0.4162577986717224) q[11], q[49];
rzx(0.0771288275718689) q[12], q[49];
rzx(0.10881084203720093) q[13], q[49];
rzx(0.45964038372039795) q[14], q[49];
rzx(0.44197893142700195) q[15], q[49];
rzx(0.4614109396934509) q[16], q[49];
rzx(0.9682124257087708) q[17], q[49];
rzx(0.9276533126831055) q[18], q[49];
rzx(0.5060545802116394) q[19], q[49];
rzx(0.07347863912582397) q[20], q[49];
rzx(0.9173610806465149) q[21], q[49];
rzx(0.6895627379417419) q[22], q[49];
rzx(0.8586465716362) q[23], q[49];
rzx(0.5256342887878418) q[24], q[49];
rzx(0.7179959416389465) q[25], q[49];
rzx(0.04517793655395508) q[26], q[49];
rzx(0.07871842384338379) q[27], q[49];
rzx(0.7379534244537354) q[28], q[49];
rzx(0.4286956787109375) q[29], q[49];
rzx(0.10952061414718628) q[30], q[49];
rzx(0.5697977542877197) q[31], q[49];
rzx(0.33375275135040283) q[32], q[49];
rzx(0.7289881110191345) q[33], q[49];
rzx(0.7214542031288147) q[34], q[49];
rzx(0.37377893924713135) q[35], q[49];
rzx(0.4208277463912964) q[36], q[49];
rzx(0.7292264103889465) q[37], q[49];
rzx(0.6526039838790894) q[38], q[49];
rzx(0.19967228174209595) q[39], q[49];
rzx(0.5596750974655151) q[40], q[49];
rzx(0.708590030670166) q[41], q[49];
rzx(0.11044764518737793) q[42], q[49];
rzx(0.5987625122070312) q[43], q[49];
rzx(0.24435943365097046) q[44], q[49];
rzx(0.6742043495178223) q[45], q[49];
rzx(0.6265973448753357) q[46], q[49];
rzx(0.03946477174758911) q[47], q[49];
rzx(0.9701914191246033) q[48], q[49];
h q[0];
