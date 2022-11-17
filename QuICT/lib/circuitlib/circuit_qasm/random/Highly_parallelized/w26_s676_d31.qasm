OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
creg c[26];
rz(3.217668804135075) q[8];
rz(1.0008298071147192) q[7];
rz(3.276024871827704) q[22];
rz(1.0807641772836698) q[15];
rz(3.087275799448039) q[11];
rz(0.5686166862741561) q[17];
rz(2.636542580695416) q[9];
rz(4.50709817534861) q[21];
rz(4.235443997869234) q[4];
cx q[5], q[12];
rz(3.437492033867331) q[23];
rz(4.894635266578666) q[14];
rz(3.315901599001975) q[0];
rz(2.937168373544628) q[25];
rz(5.646627888557418) q[20];
rz(0.47900527476077215) q[16];
rz(3.938454331310787) q[1];
rz(0.6683402765011247) q[10];
rz(5.689640655709533) q[18];
cx q[19], q[6];
rz(3.599331179078299) q[3];
rz(5.76526529837167) q[24];
rz(6.105839779348626) q[13];
rz(4.671516010553363) q[2];
cx q[13], q[24];
rz(4.3444278079255465) q[23];
rz(3.792928691211825) q[12];
rz(1.8452644332089683) q[20];
rz(0.02709609061753275) q[9];
rz(1.5848370647694507) q[8];
rz(4.0916989357369555) q[0];
rz(5.174906561943713) q[7];
rz(0.3248196664679025) q[15];
rz(5.131416085112098) q[16];
rz(5.300953606708513) q[2];
rz(2.6127793535576647) q[10];
rz(0.37125892081671896) q[6];
rz(1.1743319737542521) q[21];
rz(5.165599304587943) q[17];
rz(5.726136542121647) q[4];
rz(4.277711426208497) q[11];
rz(4.297515823096904) q[5];
rz(1.8455715404791024) q[14];
rz(1.9447591434712108) q[18];
rz(1.1463506913241182) q[3];
rz(2.456657586207485) q[1];
rz(1.0255787247340344) q[19];
rz(5.451669325990927) q[22];
rz(5.2442584221534005) q[25];
rz(5.305321370249519) q[3];
rz(3.1306604628234997) q[5];
rz(4.900543016276545) q[1];
rz(5.330416322912836) q[4];
rz(1.8928911361017466) q[10];
rz(3.644371416008752) q[18];
cx q[2], q[6];
rz(4.405399306084817) q[15];
rz(1.424673137431093) q[17];
rz(2.551537678202166) q[23];
rz(0.0662463539722499) q[14];
cx q[22], q[13];
rz(5.755805007483046) q[16];
rz(3.86835507932783) q[11];
rz(0.9226040937381035) q[7];
rz(0.17502211211563265) q[19];
rz(4.519310008472613) q[12];
rz(0.7607130329539664) q[21];
rz(3.5306874025161044) q[0];
rz(2.6898002111162236) q[24];
rz(5.455651355597032) q[20];
rz(0.9621674808684438) q[8];
rz(5.390963000813125) q[25];
rz(5.988685194571035) q[9];
rz(3.3761477937621853) q[23];
rz(4.413338735146224) q[25];
rz(2.5299084264341523) q[10];
rz(5.425038552514325) q[21];
cx q[8], q[6];
cx q[0], q[22];
rz(0.45347481658998245) q[18];
rz(6.069260668896539) q[14];
cx q[16], q[24];
cx q[13], q[9];
cx q[11], q[2];
rz(0.018147459198864922) q[19];
rz(6.019533316018024) q[5];
rz(2.5649308923410823) q[17];
cx q[15], q[7];
rz(1.9987847789724495) q[20];
rz(4.848471844483571) q[12];
rz(5.120274938648191) q[4];
cx q[3], q[1];
rz(2.6262812750130444) q[8];
rz(6.219095047890162) q[7];
rz(4.723179224464241) q[10];
rz(0.9026141444345722) q[16];
rz(3.8252584443522433) q[12];
rz(2.0749889053455037) q[25];
cx q[22], q[19];
rz(0.48114944829258655) q[20];
rz(3.639080380534558) q[15];
cx q[1], q[18];
rz(3.5927507784833024) q[23];
rz(4.634786951339709) q[6];
rz(1.3303306387863465) q[4];
rz(0.15408239157962947) q[11];
rz(1.5490674487105713) q[13];
rz(1.6744629778880313) q[2];
rz(5.853348375203298) q[21];
rz(3.1445924727598005) q[3];
rz(3.455250263028283) q[9];
rz(4.890314753600662) q[17];
rz(2.589209051563435) q[0];
rz(2.737112008778203) q[14];
rz(2.65031993141142) q[5];
rz(1.5194685291061747) q[24];
rz(5.956437267103433) q[16];
rz(2.639724576104412) q[3];
cx q[9], q[10];
rz(5.505906356543269) q[5];
rz(3.8985228322302596) q[7];
rz(3.1580943318188126) q[4];
cx q[22], q[21];
rz(0.9044472000889859) q[11];
rz(5.2261298205345375) q[0];
cx q[15], q[12];
rz(4.404146394139106) q[6];
rz(3.2711418273306343) q[20];
rz(1.0435384191967596) q[24];
rz(4.286938846141464) q[8];
rz(1.3910355438106032) q[25];
rz(0.04595643927131211) q[1];
rz(0.3293905077600433) q[13];
cx q[18], q[14];
rz(1.721929757235678) q[2];
rz(1.365232983962646) q[23];
rz(2.891829379349123) q[19];
rz(1.5850322150699656) q[17];
rz(2.924755833906831) q[24];
rz(3.7523642525491745) q[0];
rz(4.023428735876258) q[2];
rz(1.039980992554678) q[13];
rz(4.613983065551482) q[8];
rz(2.991413214799458) q[14];
cx q[20], q[7];
rz(3.4078315705965103) q[25];
rz(4.423665995099885) q[15];
rz(3.6971921673011625) q[1];
rz(3.140390831701181) q[11];
rz(1.4787314293050242) q[16];
rz(5.103028742812583) q[23];
rz(4.576158819614489) q[9];
rz(3.9372582055313035) q[12];
rz(4.926891731802666) q[5];
rz(2.764074366058084) q[3];
rz(6.064524048253295) q[19];
cx q[22], q[21];
rz(1.6756305437387016) q[4];
rz(4.407201224392974) q[6];
rz(1.838225709271514) q[17];
rz(3.382331652671802) q[18];
rz(5.324606161049059) q[10];
rz(5.378179017740857) q[21];
cx q[11], q[8];
cx q[9], q[1];
rz(0.1098106832375667) q[5];
rz(5.192318859551179) q[17];
cx q[12], q[25];
rz(2.2239747349823835) q[0];
cx q[4], q[18];
rz(4.714289268934751) q[2];
rz(3.204785313247978) q[19];
rz(5.112511037688307) q[22];
rz(2.5799856107264367) q[6];
rz(5.234149674841133) q[10];
rz(4.539754850520482) q[24];
rz(3.4736580034427664) q[16];
rz(4.269213657637955) q[23];
rz(5.286006952546048) q[14];
rz(6.197693354365431) q[7];
rz(4.762236339299519) q[20];
rz(2.521417032185305) q[13];
cx q[3], q[15];
rz(5.264283602609564) q[20];
rz(0.566988392590981) q[10];
rz(4.766768280256419) q[23];
rz(0.815993216584277) q[17];
rz(5.935879599652729) q[13];
rz(6.0370172369052035) q[4];
rz(4.826246935754629) q[3];
rz(0.20691746502805003) q[11];
rz(3.4523655756590896) q[2];
rz(4.563616939710516) q[25];
rz(4.592145781383333) q[0];
rz(1.2191918326320141) q[8];
cx q[6], q[18];
rz(0.3980492755361884) q[14];
rz(4.688465398278139) q[16];
rz(5.502744707288168) q[7];
rz(0.11046687617460266) q[19];
rz(3.4266225258630096) q[5];
rz(4.576711385331146) q[9];
cx q[1], q[24];
cx q[12], q[21];
rz(2.617248143052302) q[15];
rz(5.279467753835051) q[22];
rz(5.119373873580751) q[25];
cx q[6], q[21];
rz(1.9314325329154118) q[4];
cx q[11], q[1];
rz(0.3723601417343522) q[12];
cx q[17], q[0];
rz(2.1395853470999224) q[23];
rz(4.294438931908712) q[14];
rz(3.398156597198739) q[13];
rz(5.8517481189600185) q[22];
rz(4.836437577988718) q[24];
rz(2.875395895766972) q[15];
rz(4.098579600646349) q[20];
rz(0.9932976162023328) q[18];
rz(0.5990913905805748) q[8];
rz(5.944214680775584) q[19];
rz(5.446965351720255) q[5];
rz(6.235119251004628) q[9];
rz(0.9464653815556998) q[16];
rz(5.752777499305356) q[7];
rz(3.170871455969068) q[10];
rz(0.8095197380687005) q[2];
rz(3.009852979096678) q[3];
rz(4.0768777295641145) q[9];
rz(5.62049778962938) q[12];
rz(0.31385408419571614) q[3];
rz(5.871332671356264) q[22];
rz(0.04355798315849272) q[17];
rz(4.529984224352159) q[13];
rz(3.303312782022579) q[21];
rz(1.2315109908644297) q[8];
cx q[16], q[23];
rz(5.338168776798702) q[10];
rz(5.097071896357047) q[1];
rz(3.860261687087427) q[5];
rz(0.03374991338571054) q[4];
rz(4.790980385932247) q[15];
rz(0.9875140290007481) q[25];
rz(3.557532181069645) q[0];
rz(3.9672476974061675) q[6];
rz(5.445298574118049) q[20];
rz(0.5441115815180954) q[24];
rz(5.049284589485976) q[2];
rz(3.4914475996590184) q[18];
cx q[7], q[14];
rz(1.4461394355484136) q[19];
rz(5.421654307268885) q[11];
cx q[11], q[6];
cx q[17], q[24];
rz(0.3779041121708562) q[1];
rz(5.471517890840288) q[2];
rz(5.111754805164309) q[14];
rz(1.2536945893621427) q[22];
rz(6.052963098393157) q[9];
rz(2.00049178896587) q[5];
cx q[8], q[7];
cx q[25], q[20];
rz(5.7889899523327735) q[19];
rz(1.5435567928196086) q[13];
rz(2.380266009299508) q[12];
rz(1.8549569194762145) q[3];
rz(0.7893325058909804) q[0];
rz(3.5379033789411594) q[18];
rz(4.000914951108192) q[15];
rz(6.273647607806565) q[10];
cx q[23], q[16];
cx q[21], q[4];
rz(1.3724335818932563) q[24];
rz(2.9704466317615097) q[17];
cx q[18], q[12];
rz(4.73863738647045) q[10];
rz(3.044914300292517) q[9];
rz(4.735381449300665) q[8];
rz(6.166371783803719) q[15];
rz(0.36277322998387634) q[20];
rz(6.252377913019551) q[21];
rz(4.944264476683979) q[3];
rz(0.3489153065556205) q[19];
rz(3.596080294231039) q[25];
rz(0.06068548339393848) q[2];
rz(1.992217205981557) q[7];
rz(3.5051849412317067) q[0];
rz(4.888826842270091) q[11];
rz(5.012878851599013) q[1];
rz(2.4611829446957523) q[23];
rz(5.401281251520311) q[4];
rz(1.2746612703136848) q[13];
rz(2.483621757196454) q[22];
rz(3.730699641601912) q[6];
cx q[5], q[16];
rz(1.0962021452430468) q[14];
cx q[23], q[25];
rz(2.8958704351729962) q[12];
rz(2.593719166970899) q[21];
rz(3.3358872752123068) q[16];
cx q[1], q[15];
rz(3.8290586728818754) q[2];
rz(1.6600020548741) q[14];
rz(5.931153570621744) q[11];
rz(1.898687928004827) q[3];
rz(4.90112243235117) q[4];
rz(4.527881464989289) q[13];
rz(4.801139327170414) q[19];
rz(1.400799958274295) q[10];
rz(0.7517378032411061) q[0];
rz(1.564926906262973) q[6];
cx q[18], q[22];
rz(0.4856346354784839) q[5];
rz(0.33487997017949095) q[20];
rz(2.9635262689672532) q[9];
cx q[7], q[8];
rz(3.3977861116211043) q[24];
rz(0.9675714707086513) q[17];
rz(3.819429603856562) q[1];
rz(0.5456709206779983) q[20];
rz(3.984388094807303) q[14];
rz(1.2904547220229041) q[2];
rz(0.12225736211698851) q[5];
rz(1.6666001799965566) q[0];
rz(1.8959398677865142) q[4];
rz(3.0060321782982187) q[25];
rz(3.265898898208936) q[17];
rz(4.353850126091346) q[23];
cx q[11], q[16];
rz(2.894812027935441) q[9];
rz(3.536466271056853) q[10];
rz(4.15298459424949) q[21];
rz(1.8038414197309454) q[3];
rz(3.7736893630189896) q[18];
rz(3.2037785493894853) q[12];
cx q[7], q[22];
rz(2.5586586321581835) q[6];
rz(0.8297696938335253) q[19];
rz(3.286742554276466) q[13];
rz(4.414240633800311) q[15];
rz(2.8074226062626475) q[24];
rz(0.9414147905960837) q[8];
rz(1.299392085000301) q[5];
rz(6.225008163390091) q[4];
rz(3.3823758987108477) q[24];
rz(0.7083697789817432) q[2];
rz(2.7182626924858844) q[15];
rz(5.989697328989705) q[12];
rz(3.2786728613578293) q[9];
rz(2.7503104078150225) q[3];
cx q[16], q[7];
rz(2.817031351468646) q[8];
rz(4.984971983235339) q[14];
rz(5.2338142755548835) q[19];
rz(3.349528989488224) q[0];
rz(1.6175267078621363) q[23];
rz(3.6638901476102537) q[17];
rz(5.662765388661182) q[18];
rz(0.7585843857664996) q[13];
cx q[6], q[25];
rz(6.1820521071939805) q[11];
rz(5.069098995603157) q[10];
rz(5.126705035500636) q[20];
rz(3.660983294014434) q[21];
cx q[1], q[22];
rz(1.9019357150705243) q[25];
cx q[3], q[24];
rz(5.5497489300178735) q[14];
rz(4.993728391970664) q[5];
rz(4.305871737977515) q[21];
rz(2.9293097557932706) q[6];
rz(4.934664978963723) q[16];
rz(3.125706975692947) q[4];
rz(2.9251329299170896) q[20];
rz(2.0552396526989214) q[7];
rz(6.280899844366259) q[10];
cx q[17], q[1];
rz(1.3959168180766675) q[18];
rz(2.146238657294868) q[11];
rz(0.10089489990044573) q[19];
rz(1.9288671900918746) q[9];
rz(0.42866079138985613) q[22];
cx q[15], q[0];
rz(3.76273361529704) q[13];
rz(2.3644234816575356) q[12];
rz(2.6569511990001464) q[2];
rz(5.635906586283251) q[8];
rz(0.6499878133486688) q[23];
rz(5.446000895501022) q[13];
rz(1.586535970241614) q[7];
rz(5.313382666483687) q[8];
rz(4.561768693994042) q[15];
rz(3.532012169095273) q[9];
rz(3.6729546301956018) q[0];
rz(0.4969468628658945) q[6];
rz(0.49331306964369553) q[16];
rz(5.926604681759521) q[12];
rz(3.64945235929696) q[19];
rz(1.1298297942591944) q[24];
rz(4.700168413505656) q[3];
rz(4.411687739195939) q[11];
rz(5.296788487380317) q[18];
rz(5.902136388764494) q[22];
rz(5.78086782077583) q[2];
rz(2.6647730975511283) q[5];
cx q[25], q[1];
rz(3.112212040496784) q[21];
rz(2.9638860880016473) q[20];
rz(4.695395149201047) q[17];
rz(0.9167618162736803) q[23];
rz(4.310617060969066) q[10];
rz(1.8456913655237892) q[4];
rz(2.9886455984062223) q[14];
rz(5.0966878464919505) q[4];
rz(2.795710111681842) q[7];
rz(1.0574801969078063) q[17];
rz(5.933890509314869) q[9];
rz(3.0266430673177402) q[22];
rz(0.617149145415365) q[14];
rz(6.110037156273052) q[19];
rz(4.373203208603241) q[18];
rz(3.9591718521344657) q[15];
rz(3.271729480344817) q[2];
cx q[21], q[6];
rz(4.730511265885752) q[23];
rz(4.526383025866627) q[3];
rz(4.995132711589153) q[20];
rz(1.5291990572361183) q[25];
rz(1.1448822087607018) q[13];
rz(0.46236461901152065) q[1];
rz(3.9810904112796437) q[24];
cx q[11], q[16];
rz(3.56607371924094) q[8];
rz(3.285375401605992) q[12];
rz(1.3661290915053705) q[5];
rz(2.3406234268379698) q[10];
rz(5.756702832717834) q[0];
cx q[11], q[20];
rz(1.8628209642183726) q[14];
rz(4.849482334335668) q[5];
rz(3.558207175743358) q[23];
rz(4.516095215990082) q[19];
cx q[4], q[22];
rz(2.9496164534981424) q[10];
rz(2.2240995755133257) q[6];
rz(4.620030513918292) q[1];
rz(0.5680038583950413) q[3];
rz(0.8602618903236349) q[7];
rz(4.899808074301775) q[0];
rz(0.9612767964723984) q[16];
rz(3.682557803774622) q[12];
rz(0.9060829492241719) q[15];
rz(3.1290624115018284) q[25];
cx q[2], q[13];
rz(5.318339796836259) q[8];
cx q[9], q[24];
rz(4.050061313997815) q[17];
cx q[21], q[18];
rz(5.052489442223933) q[7];
cx q[16], q[0];
rz(4.147001631156407) q[10];
rz(3.0186469810077003) q[8];
rz(2.6662709699155442) q[15];
rz(1.5240786562170399) q[11];
rz(0.9626814616798739) q[22];
cx q[12], q[19];
rz(3.672742367709832) q[6];
rz(1.8945354994446404) q[20];
rz(2.103034371816586) q[2];
cx q[14], q[1];
rz(0.9112329059751333) q[18];
cx q[4], q[9];
rz(5.324499951174068) q[17];
rz(1.7149530314675738) q[5];
rz(5.189733318657012) q[25];
rz(1.3043865531539927) q[21];
rz(0.7366723830068093) q[13];
cx q[23], q[24];
rz(0.2936063293176139) q[3];
rz(5.012611910503356) q[19];
rz(0.48370955820754896) q[15];
rz(0.1900244690851715) q[24];
rz(4.204741513792506) q[20];
cx q[23], q[2];
rz(1.8115633296767881) q[10];
rz(2.1156327298929107) q[17];
rz(1.6607325867006968) q[9];
rz(4.65840989359405) q[7];
rz(5.646095677125983) q[21];
cx q[5], q[3];
cx q[4], q[6];
cx q[25], q[8];
rz(1.4299944691086965) q[18];
rz(5.9209990802857675) q[14];
rz(1.9115926673322898) q[11];
rz(3.3771452139035127) q[16];
cx q[13], q[22];
rz(5.433626197066233) q[1];
cx q[0], q[12];
rz(0.3682493361410648) q[24];
rz(2.8767817544011045) q[6];
rz(1.3796872808006062) q[18];
rz(1.5592897210714034) q[16];
rz(0.06917535890303789) q[19];
rz(6.149279980182805) q[7];
rz(2.3480009784792846) q[3];
rz(3.9100977268822334) q[20];
rz(3.1215128674580694) q[17];
rz(0.28912115442269587) q[12];
rz(0.0646180875471083) q[23];
cx q[8], q[4];
rz(4.31283241379264) q[10];
rz(4.476948304213305) q[0];
rz(1.3703182626140669) q[15];
cx q[25], q[21];
cx q[22], q[14];
cx q[5], q[13];
rz(1.5180814765451036) q[11];
rz(4.179749261697155) q[2];
cx q[1], q[9];
rz(6.020639608090295) q[22];
rz(0.8310086354651723) q[16];
rz(5.983247348231591) q[14];
rz(3.5974903574918407) q[13];
cx q[10], q[5];
rz(0.15913124643239016) q[23];
rz(0.6465886857479384) q[19];
rz(1.7513239752003684) q[20];
rz(6.222238950530857) q[25];
cx q[0], q[1];
rz(4.628647204016922) q[8];
rz(0.10594097191422289) q[6];
cx q[2], q[21];
rz(1.6461255709278042) q[4];
rz(5.210276987511856) q[11];
rz(4.570422759971058) q[3];
rz(5.114624710822386) q[12];
rz(3.266806458966058) q[24];
cx q[17], q[15];
cx q[18], q[7];
rz(3.1835355290163827) q[9];
cx q[22], q[0];
rz(0.6760113306017743) q[2];
rz(2.9956948158978918) q[18];
cx q[21], q[24];
rz(0.3275989137361627) q[25];
rz(1.783361490544783) q[3];
rz(4.964209113582665) q[1];
cx q[5], q[17];
cx q[4], q[6];
cx q[13], q[20];
rz(5.2581993218874095) q[8];
cx q[10], q[9];
rz(0.0063266261943364125) q[14];
rz(4.624421968774316) q[11];
rz(4.124355620243562) q[7];
rz(2.861445425815097) q[12];
rz(3.0553517356155124) q[15];
rz(2.911246265084046) q[16];
rz(4.534952555307594) q[19];
rz(3.7366250332818054) q[23];
rz(2.1136253160464515) q[13];
rz(4.471245838920056) q[3];
rz(5.801580283023231) q[18];
cx q[16], q[0];
cx q[8], q[21];
rz(0.6840208408615237) q[1];
rz(0.7767318148181768) q[19];
cx q[14], q[2];
rz(5.5075183030898) q[17];
rz(0.4895415874184214) q[7];
rz(3.7500649223593636) q[6];
cx q[22], q[10];
rz(5.33355344129239) q[24];
rz(3.442263925858958) q[25];
rz(4.1282636985494445) q[11];
rz(5.301112570886349) q[23];
cx q[12], q[5];
rz(5.634334978993748) q[9];
rz(0.3041636466176238) q[4];
rz(0.24382318940249276) q[15];
rz(5.393821668416076) q[20];
rz(2.9060632532459727) q[5];
rz(6.076994983981943) q[13];
rz(5.699347786866151) q[7];
rz(4.6967891002763285) q[14];
cx q[4], q[1];
rz(5.21972207353027) q[9];
rz(5.5785262492827945) q[16];
rz(1.625148685502584) q[8];
rz(1.2833324835227422) q[10];
rz(5.908123423179787) q[12];
cx q[21], q[25];
rz(3.386820594055444) q[17];
rz(0.8644137156338052) q[3];
rz(1.717211649702163) q[22];
rz(3.593921381830576) q[6];
cx q[0], q[19];
rz(5.137687041901303) q[20];
cx q[18], q[15];
cx q[24], q[23];
rz(2.329498005410189) q[11];
rz(1.517215076729334) q[2];
rz(3.9292548564442504) q[19];
rz(1.7057378382216946) q[11];
rz(5.253916737483878) q[16];
cx q[4], q[5];
cx q[12], q[10];
rz(4.512855657236834) q[8];
cx q[20], q[24];
cx q[22], q[1];
rz(4.327993292648504) q[15];
rz(1.0848049302908462) q[23];
rz(1.373784715885745) q[13];
cx q[7], q[18];
rz(1.6913458515941198) q[0];
cx q[14], q[2];
rz(1.962204747833149) q[25];
rz(0.47056310759725317) q[9];
rz(3.6761011912658836) q[3];
rz(5.722357051809605) q[17];
rz(5.033992861533211) q[21];
rz(5.830867276128175) q[6];
cx q[6], q[5];
rz(4.671633943246275) q[1];
rz(5.738351778898636) q[14];
rz(0.765427597531156) q[24];
rz(0.17699267726264206) q[3];
rz(0.7933893498109316) q[18];
rz(2.931403780218915) q[13];
rz(4.766856126956002) q[15];
rz(0.30678711635897943) q[23];
rz(4.7196406281003265) q[19];
cx q[16], q[2];
rz(0.3394582293267794) q[21];
rz(2.31873338417318) q[12];
rz(1.974887456075807) q[17];
rz(4.710512000095384) q[9];
rz(5.361332440800033) q[25];
rz(3.53602824635989) q[10];
cx q[4], q[11];
rz(3.6905461325201117) q[8];
rz(4.083437808625346) q[22];
rz(5.804624792051247) q[20];
cx q[0], q[7];
rz(4.99836011571605) q[11];
rz(2.158198349361398) q[9];
rz(0.6528438252497715) q[2];
rz(6.152215778087338) q[8];
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