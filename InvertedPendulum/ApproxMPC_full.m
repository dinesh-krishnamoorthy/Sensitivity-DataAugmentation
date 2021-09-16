function [Y,Xf,Af] = ApproxMPC_full(X,~,~)
%APPROXMPC_FULL neural network simulation function.
%
% Auto-generated by MATLAB, 15-Sep-2021 20:09:17.
% 
% [Y] = ApproxMPC_full(X,~,~) takes these arguments:
% 
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = 2xQ matrix, input #1 at timestep ts.
% 
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = 1xQ matrix, output #1 at timestep ts.
% 
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;-5];
x1_step1.gain = [0.318309886183791;0.2];
x1_step1.ymin = -1;

% Layer 1
b1 = [5.0337918378890433857;3.0398641420441605554;2.9360739222680636296;-0.49452046626245593774;-0.97142950447563947147;3.1221759528171024733;2.6984329569084102296;3.6307130257064472723;-3.7065397410077545182;-5.0552188277464118471];
IW1_1 = [-3.1888494174809873805 -2.049882164618223257;-2.6777709510661140691 2.8947116277563047504;-5.330534133857177892 -1.0235009416545715055;-1.8739683114872121017 -0.54188696264321911666;0.79874178959310704329 -5.8240874727380038678;3.5316042491735619713 -4.1383949769489651871;2.7502485845229216466 -3.6617147855700338255;-0.6499681516731798947 -2.9157806249515045849;0.47057049667393074932 -4.4112578784850589031;-3.1475358420198280562 -1.4802805738830036031];

% Layer 2
b2 = [2.2667669563235888219;-1.0196013712627043368;-1.5016510977104509106;0.98519316196156780929;-0.63870897294501149943;0.61817384449317935857;-1.3912834577549679782;-1.4099922072258230799;0.971417191143167158;1.3111439402859017545];
LW2_1 = [-0.85900430303587549385 -0.034121945778138623728 -1.1794069733447365422 -0.944102816382455301 0.001094441764490338689 0.23572996404164101025 0.31757220011827824724 -0.73242669041646668582 0.059286370125219263472 -0.52109470900521015491;-0.95232378156526753532 0.37999220198964323014 -0.1166651859757596027 0.04566597124379960515 -2.7977544545945827181 2.7251861750391110917 -1.8478891818903988575 -0.72170295501595738141 0.58454154940656521955 1.5570521595947832694;0.017685829425880487259 -1.4626630127890947186 0.11968252128846926241 -0.43567958425886843532 1.3892395222432447621 0.10649989624539545063 0.83827283705257993329 -0.52482902832562161599 1.7219106748380765826 -3.6409258767976453441;1.2930980968975152035 -0.46047397550723340176 1.2230479597918928469 -0.26287678281240944855 0.20362043211504915474 -0.29733339443137679625 -0.51573662171390377562 -3.2982648594802808084 0.078915910771338027008 -0.11569241987163719765;-0.69399322499478921511 1.2759064367058146061 0.80273526468809752732 0.65623622134160408681 0.46766041664870444672 2.1600903859710149924 1.9161021581631638711 -0.20228099567005985993 1.3040563624758481076 1.7234044779874118092;1.7197339717644395396 0.15645719916744724176 3.2493889555943540337 -0.45146041350828286509 -0.062685929977520524203 -0.67146808267143742732 1.325070085347483051 -0.0018125079502654744754 -1.0406388326193978244 -0.072143301495631390052;-0.20783034183484100366 -0.5099343276156669269 0.37168681639197137523 0.84285319591151008911 0.082599155015388364687 0.88389118268517707921 1.2609116952908792086 0.83747693772545162005 0.060909619196784912931 0.48514069777902901759;-1.1200381075244947304 -1.88692771597957476 1.5856430483307377699 -0.45937234650144564974 0.12898020206990151726 -0.024014553263718416143 -1.4673693533621359197 -0.36888158187492592344 0.50196902970457135673 1.1144306409208732944;1.82962989948190069 -0.76110565069084246126 0.61713188229926441153 0.83637640252056433265 0.64538336932479467567 -0.86570889767262360071 0.26596425654834987551 2.1107934800449488222 -1.3304631924747105653 0.85957039331431595031;0.038454872184164205773 2.2291064749529723166 0.076904740371117205622 -0.80279614090379247671 0.59810811565270738388 -0.31091010501247834386 1.9702734015640672727 0.37766757207205109426 -0.40573205495205455495 1.3117064595263840943];

% Layer 3
b3 = [-2.1675985554971251013;2.3324166475603891335;-2.0890304912300319273;-0.18074091646814280376;-1.2743439699318008795;1.3436456250764234888;1.0341825576966217071;-2.4394571711278123516;1.0248969650888772875;2.6597705051770850204];
LW3_2 = [1.170868481999707722 1.1195712877583856582 -0.17788793692342563801 0.5780259043510735939 1.3577937776337762887 1.8763617041186280154 -1.8014370600032838521 -1.0240025396658940515 0.30766857133571123839 -1.4592870373715536303;0.80097507034866621822 -0.71118609373864161149 0.030256132427685357433 0.38426541627312210681 0.3917219847909781949 2.4800986611582338526 -1.5227443674000498142 1.4380745010488511504 0.81754866997682595908 0.46313158962944916963;-0.55536069184048431069 -1.1576390892200563165 1.6432768369800674346 0.23257671156133413737 2.1928271762255748811 -0.42893138360186550129 -0.13480693237979199384 1.3373751573264667236 -1.3705039947385884158 0.32622187217604742271;0.75734294058630269308 1.100104770537537302 -0.83361594959638329883 2.1196339577804725884 0.081546423356164521312 0.42652118758244739194 -0.080663827700122106568 0.44599427556838999731 0.27293116558002983307 0.1040603409593013895;1.5792853689996377842 1.4422801927093895635 0.74483283951110057508 -0.13736484933973483469 1.3403290205338858065 -0.10619475725775234798 2.0691884112479228541 0.2826354859232356187 -0.38524207864193493744 0.17400037129064641062;0.44914241399708659808 3.1389834539782466294 -0.16835074205948974413 0.49866614906729700429 -0.40988083434386107795 0.070216399888881494973 -0.19253226548673443852 -0.13590937314379758494 -0.6110451582546592153 0.75881875725513014963;0.69746294728398949836 -0.85160990554809035125 0.036098253373320253468 -0.67236640818754767057 -0.11216906697989378894 0.74126858992760102929 -0.95350712369295176352 0.19289798310325861896 0.63973746992822122515 0.78147424513773822152;-2.5272597899248308018 -3.778410536170396572 -0.033148674553344678606 1.7171197881925810425 -1.3643195820898956772 1.1805053780147429787 -0.58913896503704643592 -0.43427539851321694808 1.1657976020459375199 -0.95567765901093748582;1.3562536826395503731 0.50580465236405758311 -0.0019563970930825563301 0.18807087357240795389 -0.09421908018396879525 0.048910411494490289852 -0.37009769081676430202 0.6476467662383701418 -1.9272961321829316095 0.52419790007865796344;0.31437995266948165485 -0.44232798824221253309 -1.3692763180956728508 -0.36226273701620104184 0.9085530717645023957 0.74933784449632745428 0.53969837686628074902 -0.44926080682291025248 -0.040116781336232996968 0.29635637397232422741];

% Layer 4
b4 = [0.85806723607321944147;1.4923528453768111213;-1.1191297914028890403;1.1246762034008741526;-1.0291107292675918394;1.048256033495132078;0.74837626256605105368;0.38586344120777654743;-1.1414661555849701102;2.8463082529905694429];
LW4_3 = [1.5359869772149223888 -1.5611575297549520336 0.17686853542332520162 0.3315169199392684618 -1.176193016687039572 -1.7518915170512383561 -1.3622425465221665863 0.77833212842676802179 -1.116933985494898085 -0.89384087411398105427;-0.012974781116650053667 0.60007461058370503704 -0.69549804375179768989 -0.84630163013710668629 0.030482874460285257906 0.57744272710271260518 0.064847146891466839191 0.70512340244968174652 0.73014513235872735564 0.70357107528964735188;-0.63428749298932318723 -0.2879545046109867612 -1.313788859417554411 -1.026596336978132129 1.3868240195154646344 0.65790689588813522359 0.54523170745814442117 0.35613353284758414441 -0.13185294187396268129 -0.83347341308980371632;-1.134750129856067824 -2.1669335249664589682 -0.045328438724287323236 0.54178640784789455775 -0.012319066241257227781 -0.78196763891346154907 0.69639498668265431558 -0.013407897941780985013 -0.37492810234963203087 0.95351517951177378052;-0.89851330188386924203 0.047279095663091484647 1.1596606174566967962 -0.45610345265801549575 -0.92119566338646530212 0.82687561422198174732 -0.65269573695969385074 -1.6328712933075144598 1.7823726394205876566 -1.1226503618043661703;1.4409405657307068349 1.179782091383675624 -0.73984270073400326684 -1.5227653186430130106 -0.31667316536595108145 0.26578642030904503502 1.5273692496180040212 -0.048606746055147101448 -0.97278039454137910091 0.62755751333841391926;0.056852567717686215698 0.95951823597594476656 0.029832183649165976147 0.41131700931927944742 -1.6538263793974461446 -0.20742467291766103488 -0.18054946461684875936 -0.58533282798776253841 0.58138644173319320441 1.026352004813833485;-0.23321259822229276071 -0.30388943910560067829 -0.99137403740963470078 -0.17858371752316284931 0.74973492064926128275 -0.093669332450233483423 -0.43029440397962853693 0.020091794109432153936 2.4329228790215817746 0.43701458557845695996;1.3601729027623694623 0.67436781735388717784 1.0559221642659135654 -0.028237624572236406928 -2.250934224395118477 0.6360942973945898915 0.26128115160178178034 0.3707440927978527534 -0.70349707085583479405 -0.64292953269229591307;1.7469138552665683939 1.4859499694341491427 1.3823668147392385031 0.71391126475685207087 -0.11031961138454371429 0.90563092716304904872 0.75630237225086449637 -0.58069270518586768581 1.0966745548605123517 0.60006109431780174646];

% Layer 5
b5 = [2.2344069578052012659;-1.0218998963634655652;0.44861899028416418744;0.86775029224821642604;0.053432852069192279809;-0.79625096491317015612;1.1628129743835649013;0.4289889441705493911;-1.6199313265990498145;1.7431886540940240149];
LW5_4 = [-1.3292483262702425861 0.2800933204060524484 -1.6163049894470249424 -1.0401870665664476601 0.86377461881680916989 -0.58292292928085776005 -0.5951063674930221481 1.6149260764507140653 0.81794403882369881575 0.27574950841846485039;0.68412758813713292483 -0.12124765371269172198 -1.0099523790624878039 0.54724852665497691895 0.37941205389668236636 -0.53847368170571263679 0.28601373684257708208 -0.7637551301140780291 -0.38007382602438832819 -0.90920620081761138742;-0.050506148689476816627 -1.15254047553599559 -1.5543881700922013156 -0.96043340352772577084 -0.95092879904770444277 -1.1638329224084031921 0.065835918278103366941 -0.095107741488969138599 0.43187381480465114603 0.49919233802772872144;-0.95061843792910072359 0.72134906900459916379 -1.2644835511921534987 -1.5999194758748753742 -0.66773054390904063027 -0.050736350095885819622 -0.76353214230904276061 0.68071740315332351923 0.34181345058906548173 0.84618868365498034478;1.164320182769153611 0.88018023430234060278 -1.0358290153907032671 0.27019758153037726212 0.14778639624742223591 -0.020197253787814607034 0.52078426013661804905 0.185323019640942388 0.76576338455729486832 -1.2560586904874861958;-1.3209742835741458133 -0.12400022398434766557 -0.074165843619713414658 0.14124369335713121698 0.1973833520816162812 -0.38935722887514373491 0.84833478206691648182 1.2350361749754485974 -0.31374850629362222021 -0.33894130384867210415;1.1523452082997953649 0.042230294779557316087 -0.80235301739345554761 -1.6317737296098480115 1.1682980956604136491 0.94057005973641905072 0.31005993414360949467 -0.10034720343974512513 1.3801966565758085892 0.8791460866114961803;1.1385486623228380232 0.2789211583979890885 0.99043020170577289463 -0.99647608927079667218 -1.6237541826896553232 0.014939604119234069224 -0.58468167298925044495 -1.0264650166127999853 1.461008504346768655 -0.69052181541508639295;-0.23403495113528050009 -1.1635844424849308698 -0.17763379615994764893 -0.12580896790464540791 -0.60480962685907735388 -0.60129604906712474932 -0.030769864094591526521 -0.038156333744493960147 0.24780529499563469509 -0.90594011902185789253;0.83986493895334091153 -0.32602049153736706311 -0.013600959371490611849 -0.37313726883603509643 1.0555453824079650271 0.16170132447875468218 1.0726602348608984538 0.36471171520011641531 0.10823035294068297785 0.13691696239014705672];

% Layer 6
b6 = 0.23454481325847129081;
LW6_5 = [-0.12834963948894340113 0.61790453562704417134 -0.14763840211180689899 -0.12190099733232530321 -0.18572564502181568291 -0.080839924878797403407 0.16171805263378510875 -0.081431114968442658286 -0.6665641909373127838 -0.96699886418387104303];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 0.0194062520675682;
y1_step1.xoffset = -51.5297851701723;

% ===== SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if ~isCellX
  X = {X};
end

% Dimensions
TS = size(X,2); % timesteps
if ~isempty(X)
  Q = size(X{1},2); % samples/series
else
  Q = 0;
end

% Allocate Outputs
Y = cell(1,TS);

% Time loop
for ts=1:TS

    % Input 1
    Xp1 = mapminmax_apply(X{1,ts},x1_step1);
    
    % Layer 1
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*Xp1);
    
    % Layer 2
    a2 = tansig_apply(repmat(b2,1,Q) + LW2_1*a1);
    
    % Layer 3
    a3 = tansig_apply(repmat(b3,1,Q) + LW3_2*a2);
    
    % Layer 4
    a4 = tansig_apply(repmat(b4,1,Q) + LW4_3*a3);
    
    % Layer 5
    a5 = tansig_apply(repmat(b5,1,Q) + LW5_4*a4);
    
    % Layer 6
    a6 = repmat(b6,1,Q) + LW6_5*a5;
    
    % Output 1
    Y{1,ts} = mapminmax_reverse(a6,y1_step1);
end

% Final Delay States
Xf = cell(1,0);
Af = cell(6,0);

% Format Output Arguments
if ~isCellX
  Y = cell2mat(Y);
end
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
  y = bsxfun(@minus,x,settings.xoffset);
  y = bsxfun(@times,y,settings.gain);
  y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
  x = bsxfun(@minus,y,settings.ymin);
  x = bsxfun(@rdivide,x,settings.gain);
  x = bsxfun(@plus,x,settings.xoffset);
end
