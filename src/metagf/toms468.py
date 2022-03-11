import numpy as np


def quad(func, a, b, epsil, args =()):
    
    '''
    
    quad
    
    This subroutine attempts to calculate the integral of func(x)
    over the inverval *a* to *b* with relative error not
    exceeding `epsil`.
    The result is obtained using a sequence of 1, 3, 7, 15, 31, 63,
    127, and 255 point interlacing formulae (no integrand
    evaluations are wasted) of respective degrees 1, 5, 11, 23,
    47, 95, 191 and 383.  The formulae are based on the optimal
    extensions of the 3-point gauss formula.  Details of
    the formulae are given in 'the optimum addition of points
    to quadrature formulae' by T.N.L. Patterson, Maths. Comp.
    Vol 22, 847-856, 1968.

    Parameters
    ----------

    func : function
        A Python function or method to integrate. If `func`takes many 
        arguments, it is integrated along the axis corresponding to the
        first argument.
    a : float
        Lower limit of integration
    b : float
        Upper limit of integration
        
    args : tuple, optional
        Extra arguments to pass to `func`
    epsil : float, optional
        Relative accuracy required.  When the relative
        difference of two successive formulae does not
        exceed `epsil` the last formula computed is taken
        as the result.

    Returns
    -------
    result : array 
        This array, which should be declared to have at
        least 8 elements, holds the results obtained by
        the 1, 3, 7, etc. point formulae.  The number of
        formulae computed depends on `epsil`.
    k : integer
        result[k] holds the value of the integral to the
         specified relative accuracy.
    npts : integer
        Number integrand evaluations.
    icheck : integer
        On exit normally icheck = 0.  However, if convergence
        to the accuracy requested is not achieved, icheck = 1
        on exit.
    Abscissae and weights of quadrature rules are stacked in
    array `p in the order in which they are needed.
    
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    10NOV19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      translated from toms468 Fortran routine
    '''

    if not isinstance(args, tuple):
        args = (args,)
        
    # check the limits of integration: \int_a^b, expect a < b
    flip, a, b = b < a, min(a, b), max(a, b)

    
    
    p = np.zeros(381, dtype='float64')
    funct = np.zeros(127, dtype='complex128')
    result = np.zeros(8, dtype='complex128')
 
    p = np.array([ \
       0.77459666924148337704E+00,0.55555555555555555556E+00, \
       0.88888888888888888889E+00,0.26848808986833344073E+00,\
       0.96049126870802028342E+00,0.10465622602646726519E+00,\
       0.43424374934680255800E+00,0.40139741477596222291E+00,\
       0.45091653865847414235E+00,0.13441525524378422036E+00,\
       0.51603282997079739697E-01,0.20062852938698902103E+00,\
       0.99383196321275502221E+00,0.17001719629940260339E-01,\
       0.88845923287225699889E+00,0.92927195315124537686E-01,\
       0.62110294673722640294E+00,0.17151190913639138079E+00,\
       0.22338668642896688163E+00,0.21915685840158749640E+00,\
       0.22551049979820668739E+00,0.67207754295990703540E-01,\
       0.25807598096176653565E-01,0.10031427861179557877E+00,\
       0.84345657393211062463E-02,0.46462893261757986541E-01,\
       0.85755920049990351154E-01,0.10957842105592463824E+00,\
       0.99909812496766759766E+00,0.25447807915618744154E-02,\
       0.98153114955374010687E+00,0.16446049854387810934E-01,\
       0.92965485742974005667E+00,0.35957103307129322097E-01,\
       0.83672593816886873550E+00,0.56979509494123357412E-01,\
       0.70249620649152707861E+00,0.76879620499003531043E-01,\
       0.53131974364437562397E+00,0.93627109981264473617E-01,\
       0.33113539325797683309E+00,0.10566989358023480974E+00,\
       0.11248894313318662575E+00,0.11195687302095345688E+00,\
       0.11275525672076869161E+00,0.33603877148207730542E-01,\
       0.12903800100351265626E-01,0.50157139305899537414E-01,\
       0.42176304415588548391E-02,0.23231446639910269443E-01,\
       0.42877960025007734493E-01,0.54789210527962865032E-01,\
       0.12651565562300680114E-02,0.82230079572359296693E-02,\
       0.17978551568128270333E-01,0.28489754745833548613E-01,\
       0.38439810249455532039E-01,0.46813554990628012403E-01,\
       0.52834946790116519862E-01,0.55978436510476319408E-01,\
       0.99987288812035761194E+00,0.36322148184553065969E-03,\
       0.99720625937222195908E+00,0.25790497946856882724E-02,\
       0.98868475754742947994E+00,0.61155068221172463397E-02,\
       0.97218287474858179658E+00,0.10498246909621321898E-01,\
       0.94634285837340290515E+00,0.15406750466559497802E-01,\
       0.91037115695700429250E+00,0.20594233915912711149E-01,\
       0.86390793819369047715E+00,0.25869679327214746911E-01,\
       0.80694053195021761186E+00,0.31073551111687964880E-01,\
       0.73975604435269475868E+00,0.36064432780782572640E-01,\
       0.66290966002478059546E+00,0.40714410116944318934E-01,\
       0.57719571005204581484E+00,0.44914531653632197414E-01,\
       0.48361802694584102756E+00,0.48564330406673198716E-01,\
       0.38335932419873034692E+00,0.51583253952048458777E-01,\
       0.27774982202182431507E+00,0.53905499335266063927E-01,\
       0.16823525155220746498E+00,0.55481404356559363988E-01,\
       0.56344313046592789972E-01,0.56277699831254301273E-01,\
       0.56377628360384717388E-01,0.16801938574103865271E-01,\
       0.64519000501757369228E-02,0.25078569652949768707E-01,\
       0.21088152457266328793E-02,0.11615723319955134727E-01,\
       0.21438980012503867246E-01,0.27394605263981432516E-01,\
       0.63260731936263354422E-03,0.41115039786546930472E-02,\
       0.89892757840641357233E-02,0.14244877372916774306E-01,\
       0.19219905124727766019E-01,0.23406777495314006201E-01,\
       0.26417473395058259931E-01,0.27989218255238159704E-01,\
       0.18073956444538835782E-03,0.12895240826104173921E-02,\
       0.30577534101755311361E-02,0.52491234548088591251E-02,\
       0.77033752332797418482E-02,0.10297116957956355524E-01,\
       0.12934839663607373455E-01,0.15536775555843982440E-01,\
       0.18032216390391286320E-01,0.20357755058472159467E-01,\
       0.22457265826816098707E-01,0.24282165203336599358E-01,\
       0.25791626976024229388E-01,0.26952749667633031963E-01,\
       0.27740702178279681994E-01,0.28138849915627150636E-01,\
       0.99998243035489159858E+00,0.50536095207862517625E-04,\
       0.99959879967191068325E+00,0.37774664632698466027E-03,\
       0.99831663531840739253E+00,0.93836984854238150079E-03,\
       0.99572410469840718851E+00,0.16811428654214699063E-02,\
       0.99149572117810613240E+00,0.25687649437940203731E-02,\
       0.98537149959852037111E+00,0.35728927835172996494E-02,\
       0.97714151463970571416E+00,0.46710503721143217474E-02,\
       0.96663785155841656709E+00,0.58434498758356395076E-02,\
       0.95373000642576113641E+00,0.70724899954335554680E-02,\
       0.93832039777959288365E+00,0.83428387539681577056E-02,\
       0.92034002547001242073E+00,0.96411777297025366953E-02,\
       0.89974489977694003664E+00,0.10955733387837901648E-01,\
       0.87651341448470526974E+00,0.12275830560082770087E-01,\
       0.85064449476835027976E+00,0.13591571009765546790E-01,\
       0.82215625436498040737E+00,0.14893641664815182035E-01,\
       0.79108493379984836143E+00,0.16173218729577719942E-01,\
       0.75748396638051363793E+00,0.17421930159464173747E-01,\
       0.72142308537009891548E+00,0.18631848256138790186E-01,\
       0.68298743109107922809E+00,0.19795495048097499488E-01,\
       0.64227664250975951377E+00,0.20905851445812023852E-01,\
       0.59940393024224289297E+00,0.21956366305317824939E-01,\
       0.55449513263193254887E+00,0.22940964229387748761E-01,\
       0.50768775753371660215E+00,0.23854052106038540080E-01,\
       0.45913001198983233287E+00,0.24690524744487676909E-01,\
       0.40897982122988867241E+00,0.25445769965464765813E-01,\
       0.35740383783153215238E+00,0.26115673376706097680E-01,\
       0.30457644155671404334E+00,0.26696622927450359906E-01,\
       0.25067873030348317661E+00,0.27185513229624791819E-01,\
       0.19589750271110015392E+00,0.27579749566481873035E-01,\
       0.14042423315256017459E+00,0.27877251476613701609E-01,\
       0.84454040083710883710E-01,0.28076455793817246607E-01,\
       0.28184648949745694339E-01,0.28176319033016602131E-01,\
       0.28188814180192358694E-01,0.84009692870519326354E-02,\
       0.32259500250878684614E-02,0.12539284826474884353E-01,\
       0.10544076228633167722E-02,0.58078616599775673635E-02,\
       0.10719490006251933623E-01,0.13697302631990716258E-01,\
       0.31630366082222647689E-03,0.20557519893273465236E-02,\
       0.44946378920320678616E-02,0.71224386864583871532E-02,\
       0.96099525623638830097E-02,0.11703388747657003101E-01,\
       0.13208736697529129966E-01,0.13994609127619079852E-01,\
       0.90372734658751149261E-04,0.64476204130572477933E-03,\
       0.15288767050877655684E-02,0.26245617274044295626E-02,\
       0.38516876166398709241E-02,0.51485584789781777618E-02,\
       0.64674198318036867274E-02,0.77683877779219912200E-02,\
       0.90161081951956431600E-02,0.10178877529236079733E-01,\
       0.11228632913408049354E-01,0.12141082601668299679E-01,\
       0.12895813488012114694E-01,0.13476374833816515982E-01,\
       0.13870351089139840997E-01,0.14069424957813575318E-01,\
       0.25157870384280661489E-04,0.18887326450650491366E-03,\
       0.46918492424785040975E-03,0.84057143271072246365E-03,\
       0.12843824718970101768E-02,0.17864463917586498247E-02,\
       0.23355251860571608737E-02,0.29217249379178197538E-02,\
       0.35362449977167777340E-02,0.41714193769840788528E-02,\
       0.48205888648512683476E-02,0.54778666939189508240E-02,\
       0.61379152800413850435E-02,0.67957855048827733948E-02,\
       0.74468208324075910174E-02,0.80866093647888599710E-02,\
       0.87109650797320868736E-02,0.93159241280693950932E-02,\
       0.98977475240487497440E-02,0.10452925722906011926E-01,\
       0.10978183152658912470E-01,0.11470482114693874380E-01,\
       0.11927026053019270040E-01,0.12345262372243838455E-01,\
       0.12722884982732382906E-01,0.13057836688353048840E-01,\
       0.13348311463725179953E-01,0.13592756614812395910E-01,\
       0.13789874783240936517E-01,0.13938625738306850804E-01,\
       0.14038227896908623303E-01,0.14088159516508301065E-01,\
       0.99999759637974846476E+00,0.69379364324103267170E-05,\
       0.99994399620705437576E+00,0.53275293669780613125E-04,\
       0.99976049092443204733E+00,0.13575491094922871973E-03,\
       0.99938033802502358193E+00,0.24921240048299729402E-03,\
       0.99874561446809511470E+00,0.38974528447328229322E-03,\
       0.99780535449595727456E+00,0.55429531493037471492E-03,\
       0.99651414591489027385E+00,0.74028280424450333046E-03,\
       0.99483150280062100052E+00,0.94536151685852538246E-03,\
       0.99272134428278861533E+00,0.11674841174299594077E-02,\
       0.99015137040077015918E+00,0.14049079956551446427E-02,\
       0.98709252795403406719E+00,0.16561127281544526052E-02,\
       0.98351865757863272876E+00,0.19197129710138724125E-02,\
       0.97940628167086268381E+00,0.21944069253638388388E-02,\
       0.97473445975240266776E+00,0.24789582266575679307E-02,\
       0.96948465950245923177E+00,0.27721957645934509940E-02,\
       0.96364062156981213252E+00,0.30730184347025783234E-02,\
       0.95718821610986096274E+00,0.33803979910869203823E-02,\
       0.95011529752129487656E+00,0.36933779170256508183E-02,\
       0.94241156519108305981E+00,0.40110687240750233989E-02,\
       0.93406843615772578800E+00,0.43326409680929828545E-02,\
       0.92507893290707565236E+00,0.46573172997568547773E-02,\
       0.91543758715576504064E+00,0.49843645647655386012E-02,\
       0.90514035881326159519E+00,0.53130866051870565663E-02,\
       0.89418456833555902286E+00,0.56428181013844441585E-02,\
       0.88256884024734190684E+00,0.59729195655081658049E-02,\
       0.87029305554811390585E+00,0.63027734490857587172E-02,\
       0.85735831088623215653E+00,0.66317812429018878941E-02,\
       0.84376688267270860104E+00,0.69593614093904229394E-02,\
       0.82952219463740140018E+00,0.72849479805538070639E-02,\
       0.81462878765513741344E+00,0.76079896657190565832E-02,\
       0.79909229096084140180E+00,0.79279493342948491103E-02,\
       0.78291939411828301639E+00,0.82443037630328680306E-02,\
       0.76611781930376009072E+00,0.85565435613076896192E-02,\
       0.74869629361693660282E+00,0.88641732094824942641E-02,\
       0.73066452124218126133E+00,0.91667111635607884067E-02,\
       0.71203315536225203459E+00,0.94636899938300652943E-02,\
       0.69281376977911470289E+00,0.97546565363174114611E-02,\
       0.67301883023041847920E+00,0.10039172044056840798E-01,\
       0.65266166541001749610E+00,0.10316812330947621682E-01,\
       0.63175643771119423041E+00,0.10587167904885197931E-01,\
       0.61031811371518640016E+00,0.10849844089337314099E-01,\
       0.58836243444766254143E+00,0.11104461134006926537E-01, \
       0.56590588542365442262E+00,0.11350654315980596602E-01,\
       0.54296566649831149049E+00,0.11588074033043952568E-01,\
       0.51955966153745702199E+00,0.11816385890830235763E-01,\
       0.49570640791876146017E+00,0.12035270785279562630E-01,\
       0.47142506587165887693E+00,0.12244424981611985899E-01,\
       0.44673539866202847374E+00,0.12443560190714035263E-01,\
       0.42165768662616330006E+00,0.12632403643542078765E-01,\
       0.39621280605761593918E+00,0.12810698163877361967E-01,\
       0.37042208795007823014E+00,0.12978202239537399286E-01,\
       0.34430734159943802278E+00,0.13134690091960152836E-01,\
       0.31789081206847668318E+00,0.13279951743930530650E-01,\
       0.29119514851824668196E+00,0.13413793085110098513E-01,\
       0.26424337241092676194E+00,0.13536035934956213614E-01,\
       0.23705884558982972721E+00,0.13646518102571291428E-01, \
       0.20966523824318119477E+00,0.13745093443001896632E-01,\
       0.18208649675925219825e+00,0.13831631909506428676e-01,\
       0.15434681148137810869e+00,0.13906019601325461264e-01,\
       0.12647058437230196685e+00,0.13968158806516938516e-01,\
       0.98482396598119202090e-01,0.14017968039456608810e-01,\
       0.70406976042855179063e-01,0.14055382072649964277e-01,\
       0.42269164765363603212e-01,0.14080351962553661325e-01,\
       0.14093886410782462614e-01,0.14092845069160408355e-01,\
       0.14094407090096179347e-01 \
       ])
    icheck = 0
    k = 2
    npts = 1
#  check for trivial case.
    if ( a == b ):
#  trivial case.
        k = 2
        result[0] = 0.0 + 1j*0.0
        result[1] = 0.0 + 1j*0.0
        npts = 0
        return result, k-1, npts, icheck

#  scale factors.
    add  = ( b + a ) / 2.0
    diff = ( b - a ) / 2.0
#  1-point gauss.
    x = add
    fzero = func (x, *args )
    result[0] = 2.0 * fzero * diff
    i = 0
    iold = 0
    inew = 1
    k = 2
    acum = 0.0
    while (True):
        if ( k == 8 ):
            # convergence not achieved
            icheck = 1
            npts = inew + iold
            if flip:
                result = - result
            return result, k-1, npts, icheck
    
        k = k + 1
        acum = 0.0 + 1j*0.0
#  contribution from function values already computed.
        for j in range(1, iold+1):
            i = i + 1
            acum = acum + p[i-1] * funct[j-1]
    
    #  contribution from new function values.
        iold = iold + inew
        for j in range(inew, iold+1):
            i = i + 1
            dx = p[i-1] * diff
            x = add + dx
            f1 = func(x, *args)
            x = add - dx
            f2 = func(x, *args)
            funct[j-1] = f1 + f2
            i = i + 1
            acum = acum + p[i-1]*funct[j-1]
    
        inew = iold + 1
        i = i + 1
        result[k-1] = ( acum + p[i-1] * fzero ) * diff
#  check for convergence.
        if (np.abs(( result[k-1] - result[k-2] )) <= epsil * np.abs(result[k-1] ) ):
            icheck = 0
            npts = inew + iold
            if flip:
                result = -result
            return  result, k-1, npts, icheck


def qsub(f, a, b, epsil, args = ()):
    '''
     qsub
    
      This function routine performs automatic integration
    over a finite interval using the basic integration
    algorithm quad, together with, if necessary, a non-
    adaptive subdivision process.
      The call takes the form
        qqsub, npts, relerr, icheck  = qsub(f, a, b epsil, args)
    and causes f(x,args) to be integrated over x in (a,b) with relative
    error hopefully not exceeding epsil.  Should quad converge
    (icheck=0) then qsub will return the value obtained by it;
    otherwise subdivision will be invoked as a rescue
    operation in a non-adaptive manner.  The argument relerr
    gives a crude estimate of the actual relative error
    obtained.
      The subdivision strategy is as follows:
    Let the interval (a,b) be divided into 2**n panels at step
    n of the subdivision process.  quad is applied first to
    the subdivided interval on which quad last failed to
    converge, and if convergence is now achieved the remaining
    panels are integrated.  should a convergence failure occur
    on any panel, the integration at that point is terminated
    and the procedure repeated with n increased by 1.  The
    strategy ensures that possibly delinquent intervals are
    examined before work, which later might have to be
    discarded, is invested in well-behaved panels.  The
    process is complete when no convergence failure occurs on
    any panel and the sum of the results obtained by quad on
    each panel is taken as the value of the integral.
      The process is very cautious in that the subdivision of
    the interval (a,b) is uniform, the fineness of which is
    controlled by the success of quad.  In this way, it is
    rather difficult for a spurious convergence to slip
    through.
      The convergence criterion of quad is slightly relaxed
    in that a panel is deemed to have been successfully
    integrated if either quad converges or the estimated
    absolute error committed on this panel does not exceed
    epsil times the estimated absolute value of the integral
    over (a,b).  This relaxation is to try to take account of
    a common situation where one particular panel causes
    special difficulty, perhaps due to singularity of some
    type.  In this case, quad could obtain nearly exact
    answers on all other panels, and so the relative error for
    the total integration would be almost entirely due to the
    delinquent panel.  Without this condition, the computation
    might continue despite the requested relative error being
    achieved.
      The outcome of the integration is indicated by icheck.
      icheck = 0 - convergence obtained without invoking
                   subdivision.  This corresponds to the
                   direct use of quad.
      icheck = 1 - result obtained after invoking subdivision.
      icheck = 2 - as for icheck=1 but at some point the
                   relaxed convergence criterion was used.
                   the risk of underestimating the relative
                   error will be increased.  If necessary,
                   confidence may be restored by checking
                   epsil and relerr for a serious discrepancy.
      icheck negative
                   if during the subdivision process the
                   allowed upper limit on the number of panels
                   that may be generated (presently 4096) is
                   reached, a result is obtained which may be
                   unreliable by continuing the integration
                   without further subdivision ignoring
                   convergence failures.  This occurrence is
                   flagged by returning icheck with negative
                   sign.
    The reliability of the algorithm will decrease for large
    values of epsil.  It is recommended that epsil should
    generally be less than about 0.001.
    
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    10NOV19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      translated from toms468 Fortran routine
    '''
    
    if not isinstance(args, tuple):
        args = (args,)
    
    result = np.zeros(8, dtype = 'complex128')
    nmax = 4096
    result, k, npts, icheck = quad(f, a, b, epsil, args)
    qqsub = result[k]
    relerr = 0.0
    out = 1
    if (np.abs(qqsub) > 0.0):
        relerr = np.abs((result[k] - result[k-1])/qqsub)
        
    # check if subdivision is needed
    if (icheck == 0) :
        return qqsub, npts, relerr, icheck 

#  SUBDIVIDE.
    estim = np.abs(qqsub * epsil)
    ic = 1
    rhs = False
    n = 1
    h = b - a
    bad = 1
    
#10 
    qqsub = 0.0
    relerr = 0.0
    h = h * 0.5
    n = n + n

#  interval (a,b) divided into n equal subintervals.
#  integrate over subintervals bad to (bad+1) where trouble
#  has occurred.
    
    m1 = bad
    
    m2 = bad + 1
    
    out = 1
    
    
    
    while (True): # 50
        
        if (m1 > m2): # go to 90
        
            if out == 1: 
                #  20 integrate over subintervals 1 to (bad-1).
                m1 = 1
                m2 = bad - 1
                rhs = False
                out = 2
            
            elif out == 2:
                # 30  integrate over subintervals (bad+2) to n.
                m1 = bad + 2
                m2 = n
                out = 3
            
            elif out == 3:
                # 40  subdivision result.
                icheck = ic
                relerr = relerr / np.abs( qqsub )
                return qqsub, npts, relerr, icheck 
        else:
            
    
            #  integrate over subintervals m1 to m2.
            
            for jj in range(m1, m2+1):
                
                broke = False
                
                j = jj
                #  examine first the left or right half of the subdivided
                #  troublesome interval depending on the observed trend.
                if ( rhs ):
                    j = m2 + m1 - jj
                    
                alpha = a + h * ( j - 1 )
                beta = alpha + h
                #quad ( alpha, beta, result, m, epsil, nf, icheck, f )
                result, m, nf, icheck = quad(f, alpha, beta, epsil, args)
                comp = np.abs ( result[m] - result[m-1] )
                npts = npts + nf
                if (icheck != 1):  # go to 70
                    qqsub = qqsub + result[m]
                elif (comp <= estim): # go to 100
                    ic = isign(2, ic)
                    qqsub = qqsub + result[m]
                elif (n == nmax): # go to 60
                    ic = -np.abs(ic)
                    qqsub = qqsub + result[m]
                else:
                    bad = 2 * j - 1
                    rhs = False
                    if ( ( j - 2 * ( j / 2 ) ) == 0 ):
                        rhs = True
                    # go to 10
                    qqsub = 0.0
                    relerr = 0.0
                    h = h * 0.5
                    n = n + n
                    
                    #  interval (a,b) divided into n equal subintervals.
                    #  integrate over subintervals bad to (bad+1) where trouble
                    #  has occurred.
                    m1 = bad
                    m2 = bad + 1
                    out = 1
                    broke = True
                    break # go to 50

            # end of for jj
            if (not broke):
                
                relerr = relerr + comp
                
                if out == 1:
                    
                    #  20 integrate over subintervals 1 to (bad-1).
                    m1 = 1
                    m2 = bad - 1
                    rhs = False 
                    out = 2
                    
                elif out == 2:
                    
                    # 30  integrate over subintervals (bad+2) to n.
                    m1 = bad + 2
                    m2 = n
                    out = 3
                    
                elif out == 3:
                    
                    # 40  subdivision result.
                    icheck = ic
                    relerr = relerr / np.abs( qqsub )
                    return qqsub, npts, relerr, icheck 
                else:
                    raise Exception('Allowed values for the variable out are: 1, 2, or 3')
            
        # end if (m1 > m2)    
        
    # end while (True)
    
# end qsub    
    

def qsuba(f, a, b, epsil, args = ()):
    
    '''      
    qsuba
    
      This function routine performs automatic integration
    over a finite interval using the basic integration
    algorithm quad, together with, if necessary, an adaptive
    subdivision process.  It is generally more efficient than
    the non-adaptive algorithm qsub but is likely to be less
    reliable (see Comp.J., 14, 189, 1971 ).
      The call takes the form
        result, npts, relerr, icheck = qsuba(f, a, b, epsil, arg)
    and causes f(x, arg) to be integrated over x in (a,b) with relative
    error hopefully not exceeding epsil.  sShould quad converge
    (icheck=0) then qsuba will return the value obtained by it;
    otherwise subdivision will be invoked as a rescue
    operation in an adaptive manner.  The argument relerr gives
    a crude estimate of the actual relative error obtained.
      The subdivision strategy is as follows:
    at each stage of the process, an interval is presented for
    subdivision.  (Initially this will be the whole interval
    (a,b)).  The interval is halved and quad applied to each
    subinterval.  Should quad fail on the first subinterval,
    the subinterval is stacked for future subdivision and the
    second subinterval immediately examined.  Should quad fail
    on the second subinterval, the subinterval is
    immediately subdivided and the whole process repeated.
    Each time a converged result is obtained, it is
    accumulated as the partial value of the integral.  When
    quad converges on both subintervals, the interval last
    stacked is chosen next for subdivision and the process
    repeated.  A subinterval is not examined again once a
    converged result is obtained for it, so that a spurious
    convergence is more likely to slip through than for the
    non-adaptive algorithm qsub.
      The convergence criterion of quad is slightly relaxed
    in that a panel is deemed to have been successfully
    integrated if either quad converges or the estimated
    absolute error committed on this panel does not exceed
    epsil times the estimated absolute value of the integral
    over (a,b).  This relaxation is to try to take account of
    a common situation where one particular panel causes
    special difficulty, perhaps due to singularity of some
    type.  In this case, quad could obtain nearly exact
    answers on all other panels, and so the relative error for
    the total integration would be almost entirely due to the
    delinquent panel.  Without this condition, the computation
    might continue despite the requested relative error being
    achieved.
      The outcome of the integration is indicated by icheck.
      icheck = 0 - convergence obtained without invoking
                   subdivision.  This would correspond to the
                   direct use of quad.
      icheck = 1 - result obtained after invoking subdivision.
      icheck = 2 - as for icheck=1 but at some point the
                   relaxed convergence criterion was used.
                   the risk of underestimating the relative
                   error will be increased.  If necessary,
                   confidence may be restored by checking
                   epsil and relerr for a serious discrepancy.
      icheck negative
                   If during the subdivision process the stack
                   of delinquent intervals becomes full (it is
                   presently set to hold at most 100 numbers)
                   a result is obtained by continuing the
                   integration ignoring convergence failures
                   which cannot be accommodated by the stack.
                   This occurrence is flagged by returning
                   icheck with negative sign.
    The reliability of the algorithm will decrease for large
    values of epsil.  iI is recommended that epsil should
    generally be less than about 0.001.
    
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    10NOV19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      translated from toms468 Fortran routine
    '''
    
    if not isinstance(args, tuple):
        args = (args,)

    result = np.zeros(8, dtype = 'complex128')
    stack = np.zeros(100, dtype = 'float64')
    ismax  = 100
    
    result, k, npts, icheck = quad(f, a, b, epsil, args)
    qqsuba = result[k]
    
    relerr = 0.0
    
    if ( np.abs(qqsuba) > 0.0 ):
        relerr = np.abs ( ( result[k] - result[k-1] ) / qqsuba )
    
    # check if subdivision is needed.
    if ( icheck == 0 ):
        return qqsuba, npts, relerr, icheck 
    
    # subdivide.
    estim = np.abs ( qqsuba * epsil )
    relerr = 0.0
    qqsuba = 0.0
    is1 = 1
    ic = 1
    sub1 = a
    sub3 = b
    
    # 10 
    while (True):
        sub2 = ( sub1 + sub3 ) * 0.5
        result, k, nf, icheck = quad ( f, sub1, sub2, epsil, args )
        npts = npts + nf
        comp = np.abs ( result[k] - result[k-1] )
    
        if ( icheck == 0 ): #go to 30
            qqsuba = qqsuba + result[k]
            relerr = relerr + comp
            
        elif (comp <= estim): # go to 70
            ic = isign ( 2, ic )
            qqsuba = qqsuba + result[k]
            relerr = relerr + comp
              
         
        elif ( is1 >= ismax ):  #go to 20
            
            ic = -np.abs ( ic )
            qqsuba = qqsuba + result[k]
            relerr = relerr + comp
          
        else:#  stack subinterval (sub1,sub2) for future examination.
            stack[is1-1] = sub1
            is1 = is1 + 1
            stack[is1-1] = sub2
            is1 = is1 + 1
          
        #40    
        result, k, nf, icheck = quad ( f, sub2, sub3, epsil, args )
        npts = npts + nf
        comp = np.abs ( result[k] - result[k-1] )
          
        if ( icheck == 0 ):  #go to 50
            qqsuba = qqsuba + result[k]
            relerr = relerr + comp
            
        elif ( comp <= estim ):  #go to 80
            ic = isign ( 2, ic )
            qqsuba = qqsuba + result[k]
            relerr = relerr + comp
        else:
            # subdivide interval (sub2,sub3)
            sub1 = sub2
            continue #go to 10
        if ( is1 == 1 ): #go to 60
            icheck = ic
            relerr = relerr / np.abs ( qqsuba )
            return qqsuba, npts, relerr, icheck 
        
        # subdivide the delinquent inverval last stacked.
        is1 = is1 - 1
        sub3 = stack[is1-1]
        is1 = is1 - 1
        sub1 = stack[is1-1]
        
    #  go to 10
# end of qsuba    
      
def isign(a, b):
    '''
    Function isign(a, b) returns the value of a with the sign of b.
    
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    10NOV19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      
    
    '''     
    y = np.abs(a) * np.sign(b)
    
    return y

# end of isign      
    
def halfsine(func, a, c, epsil, args):
    """
    half_sine: computes the integral of the function func(x,args) 
    over a half-sine contour defined by the widht `a` and height `c` using Patterson
    quadrature with required precision `epsil`
    y, npts, relerr, icheck  = half_sine(func, a, c, epsil, args)

       
    Parameters
    ----------

    func : function
        A Python function or method. If `func`takes many 
        arguments, it is integrated along the axis corresponding to the
        first argument.
    a : float
        width of the half-sine contour
    c : float
        maximum height of the half-sine contour     
    epsil: float
        required precision in Patterson quadrature
    args : tuple, optional
        Extra arguments to pass to `func`
    

    Returns
    -------
    y : float
        The value of the integral
    npts : integer
        Number integrand evaluations.
    relerr : float
        The projected relative error achieved by Patterson quadrature
    icheck : integer
        On exit normally icheck = 0.  However, if convergence
        to the accuracy requested is not achieved, icheck = 1
        on exit.        
    
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    10NOV19     Ivica Stevanovic, OFCOM         First implementation
                                              
    
    """
    
    def integrand(x, func, a, c, args):
        
        """
        This subroutine defines an integrand function required for the
        computation of inverse Sommerfeld integral over the bounded interval
        (0, a) along the half-sine contour defined by the width a and 
        the height c
        
        Parameters
        ----------
        x : float
            Parameter over which the integration is performed
        func : function
            A Python function or method. 
        a : float
            width of the half-sine contour
        c : float
            maximum height of the half-sine contour     
        args : tuple, optional
            Extra arguments to pass to `func`
        
    
        Returns
        -------
        result : float
            the integrand required for the computation of the inverse Sommerfeld
            integral over the bounded interval (0, a) along the half-sine contour.
        
        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v1    10NOV19     Ivica Stevanovic, OFCOM         First implementation

        """
                                                         
    
        
        targ = np.pi * (1.0 - x) / 2.0 
        
        kro = a / 2.0 * (1 + np.cos( targ) ) + 1j* c * np.sin( targ )
        
        jcb = a * np.pi/4.0 * np.sin( targ ) - 1j * c * np.pi/2.0 * np.cos( targ )
        
        result = func(kro, *args) * jcb
        
        return result
    
    # end integrand0
    

    #  Compute the integral.
    
    y, npts, relerr, icheck = qsub(integrand, -1, 1, epsil, (func, a, c , args))
    
    return y, npts, relerr, icheck

