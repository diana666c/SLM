import numpy as np
import os
from PIL import Image
import subprocess
import time
import pygetwindow as gw

import pyautogui
from scipy import special 
import math
import cmath
from matplotlib import pyplot as plt 

#import modulador.dospi
import scipy.io


direct = "\\Users\\Admin\\Documents\\SynologyDrive\\projects\\Atomoscalientes\\scripts\\modulador\\"


def imagen_correccion(arreglo):
    correccion=np.zeros((600,800))
    for i in range(100):
        correccion[0][i]=arreglo[0][i]
    return correccion

print("\n \n -----> Bienvenido al módulo que controla el SLM. \n Para empezar utiliza la función: inicia(). \n ¡¡Recuerda al terminar tu sesión correr la función: finaliza() !!")
nlong = int(input('¿Qué long. de onda desea utilizar (1), (2), (3) ó (4)? \n 1 -> ninguna \n 2 -> 911 nm \n 3 -> 780 nm \n 4 -> 633 nm \n'))
CONST_2PI = 148

if nlong==1:
    PATRON_CORRECTOR = np.zeros(shape=(600,800),dtype=np.uint8)
    #PATRON_CORRECTOR)

else:
    # Iterar por los archivos dentro del directorio 'PatronCorrector'
    for archvs in os.listdir(os.path.join(direct, 'PatronCorrector')):
        if f'n{nlong}' in archvs:  # Verifica si 'n{nlong}' está en el nombre del archivo
            direct1 = os.path.join(direct, 'PatronCorrector', archvs)  # Crear la ruta completa del archivo
            break

    print(direct1)
    PATRON_CORRECTOR=np.array(   np.concatenate( (np.array(Image.open(direct1)),np.zeros(shape=(600,8))) ,axis=1 )   ,dtype=np.uint8)
    PATRON_CORRECTOR= imagen_correccion(PATRON_CORRECTOR)
    print(PATRON_CORRECTOR)



print('Recuerda correr finaliza() al terminar NIEVES-DISCO\n')
resp1 = 'n' #input('¿El SLM está conectado como único monitor a otra computadora (controlada por ssh)? (s/n)::')

if resp1 == 'n':
        print('OJO: Si tras elegir el tamaño del monitor vez una pantalla negra, debes hacer click en "alt" para que se actualice la posición de la ventana del Visor de Imágenes.\n')
        resp2 = input('¿Cuál es el tamaño horizontal del monitor actual? (puedes saberlo desde la configuración de pantalla de Windows)::')

# Abrir el Visor de Imágenes predeterminado de Windows para mostrar la imagen
        image_path = direct + 'mpar.bmp'  # Asegúrate de ajustar esta ruta
        subprocess.Popen([image_path], shell=True)

# Esperar un momento para dar tiempo al visor de imágenes para abrirse
        time.sleep(3.5)
        print('uno')
        # Intentar mover la ventana del Visor de Imágenes a la posición correcta
        try:
            window = gw.getWindowsWithTitle('mpar.bmp')[0]  # El nombre de la ventana puede variar según el visor
            #time.sleep(2.5)
            #print('dos')
            #window.restore() # Asegura que la ventana esté restaurada (si estaba minimizada)
            #time.sleep(2.5)
            #print('tres')
            #time.sleep(2.5)
            #window.maximize() # Maximiza la ventana para asegurarse de que esté en pantalla completa
            #time.sleep(2.5)
            #print('cuatro')
            window.moveTo(int(resp2), 0)  # Mueve la ventana horizontalmente según el tamaño
            # Usar pyautogui para presionar 'F11' (pantalla completa)
            time.sleep(2.5)
            pyautogui.press('f11')  # Activa el modo de pantalla completa
            #window.moveTo(int(resp2), 0)


        except IndexError:
            print("No se pudo encontrar la ventana del Visor de Imágenes.")
        
    #elif resp1 == 's':
        # Abrir el Visor de Imágenes predeterminado de Windows para mostrar la imagen
     #   image_path = "C:\\ruta\\a\\tu\\imagen.jpg"  # Asegúrate de ajustar esta ruta
     #   subprocess.Popen([image_path], shell=True)

      #  print('Asegúrate de haber configurado correctamente el entorno gráfico en Windows antes de ejecutar el código.')

# Llamar la función para iniciar

#-------------------------------------------------------------------------------------------------------------------------------------    
#La función grayImage toma como argumento una matriz de enteros de nVer × nHor(como la que genera blazeMat) 
# y la convierte en una imagen.
def grayImageCorr(matInt:"ndarray"):
    #matInt=(modul.PATRON_CORRECTOR+matInt) # %CONST_2PI # % es para obtener módulo
    return Image.fromarray(PATRON_CORRECTOR + matInt,'L')  # El tipo 'L' es para "8-bit pixels, black and white"
#-------------------------------------------------------------------------------------------------------------------------------------
def grayImage(matInt:"ndarray"):
    return Image.fromarray(matInt,'L')  # El tipo 'L' es para "8-bit pixels, black and white" 
#-------------------
dir4 = direct + 'imagen.bmp'

def finaliza():
    # #Images.imwrite(grayImage(ones(Int64,600,800)),dir4)
    # save(dir4,grayImage(np.zeros(shape=(600,800),dtype=np.uint8)))
    tempImg=grayImage(np.zeros(shape=(600,800),dtype=np.uint8))
    tempImg.save(dir4)
    # resp2 = input('¿El SLM está conectado a esta computadora como segundo monitor? (s/n)')
    # if resp2=='s':
    #     call(['bash',dir3])
    call(['pkill','eog'])
def monitor2(imagen:"Image"):
    if imagen.size == (800,600):
        imagen.save(dir4)
    else:
        print('ERROR: Deben ser imágenes de 800x600')
        return
    sleep(1.1) # esto es para que le dé tiempo de guardar y cambiar la imagen en eog Viewer
    return None
# Guarda imágen en negro para proyectar en modulador al iniciar.
grayImage(np.zeros(shape=(600,800),dtype=np.uint8)).save(dir4)

#La función blazeMat da un arreglo de cols×rengs enteros que representa una rejilla blaze.
#El primer argumento es el periodo (en pixeles, i.e. debe ser un entero)
#El segundo argumento es el entero que representa la fase 2π.
def blazeMat(periodo,dosPi=CONST_2PI,cols=800,rengs=600):
    reng = np.array(  (dosPi)*np.mod(  np.linspace(0,cols-1,cols, dtype=int),  periodo)/(periodo-1)  ,dtype=np.uint8)
    return np.tile(reng,(rengs,1))


#La función escalon da un arreglo de cols×rengs enteros que representa una rejilla binaria.
#El primer argumento es el periodo (en pixeles, i.e. debe ser un entero)
#El segundo(tercer) argumento es el valor que toma la cresta(valle) de la rejilla binaria.
def escalon(periodo,dosPi,fondo=0,cols=800,rengs=600):#dosPi=CONST_2PI,fondo=0,cols=800,rengs=600):
    if (fondo<0 or fondo>255 or dosPi<0 or dosPi>255 or dosPi<fondo):
        print('ERROR: debe ocurrir que 0 <= fondo <= dosPi <= 255')
        return
    else:
        if np.mod(periodo,2) != 0: #si periodo no es par
            print('WARNING: esta función está optimizada cuando "periodo" es un número par')
        reng= np.array(  fondo+(dosPi-fondo)*np.round( np.mod( np.linspace(0,cols-1,cols) , periodo)/(periodo-1) )  ,dtype=np.uint8)
        return np.tile(reng,(rengs,1))
    

#La función thetaMat da un arreglo de 800x600 cuyas entradas representan los ángulos (van de -π a π)
#El argumento th son los grados por los cuales se puede rotar el holograma
#Recuerda que la convención para matrices es invertir eje Y, por eso valores negativos quedan arriba
def thetaMat(th=0):
    x = np.tile(  np.linspace(-400,400-1,800,dtype=np.int)  ,  (600,1)  )
    y = np.transpose(np.tile(  np.linspace(300,-300+1,600,dtype=np.int)  ,  (800,1)  ))
    xp = np.cos(th*np.pi/180)*x-np.sin(th*np.pi/180)*y # rotacion
    yp = np.sin(th*np.pi/180)*x+np.cos(th*np.pi/180)*y
    return np.arctan2(y,x)


#La función thetaMatInt toma la matriz de la función thetaMat y la convierte en enteros de 8-bits
#Esta es la función que se utiliza para darle vórtice al haz para (junto con axicón) generar el Bessel
def thetaMatInt(n,dosPi=CONST_2PI,th=0):
    return np.array(  np.mod( n*(dosPi)*( 0.5 + thetaMat(th)/(2*np.pi) ) , dosPi+1)  ,dtype=np.uint8)


#La siguiente función es para calibrar el SLM
#Lo que hace es esperar a que tomes una foto (hasta ahora esto debe hacerse manualmente y de manera independiente)
#para cada nivel de la función escalón
def calibrar(inicio=0,fin=255):
    print('Una vex finalizado el análisis debes incluir el resultado en "./dospi.py"')
    print('Tras lo cual debes importar nuevamente el módulo para que se tomen en cuenta los cambios')
    for i in range(inicio,fin+1):
        monitor2(grayImage(escalon(10,i))) #Este periodo permite ver los órdenes 0,1,2 en un CCD de webcam común
        resp = input('Capturar foto para i='+str(i)+'. Enter -> siguiente (abort -> interrumpir)')
        if resp=='abort':
            print('Proceso interrumpido.')
            break
    return



##################################A partir de aqui son modificaciones a este programa incluidas por Jorge Acosta ####################################
def ang_eta(x,y,h):
    n=abs(np.sqrt((x+h)**2+y**2)-np.sqrt((x-h)**2+y**2))/(2*h)
    if n>1:
        n=1
    s=math.acos(n)
    eta=0
    if x>=0 and y>=0:
        eta=s
    elif x<=0 and y>=0:
        eta=np.pi-s
    elif x<=0 and y<=0:
        eta=np.pi+s
    else:
        eta=2*np.pi-s
    return math.degrees(eta)

def xi(x,y,h):
 return math.acosh((np.sqrt((x+h)**2+y**2)+np.sqrt((x-h)**2+y**2))/(2*h))

def Ce(m,q,z): #Se definen las funciones de Mathieu modificadas a traves de las funciones de Mathieu de primer orden
    cer=0.0
    if m%2==0:
        cer=special.mathieu_cem(m,q,90)[0]*special.mathieu_cem(m,q,0)[0]*special.mathieu_modcem1(m,q,z)[0]/\
        (special.mathieu_even_coef(m,q)[0]*(-1.0)**(m/2))
    else:
        cer=special.mathieu_cem(m,q,90)[1]*special.mathieu_cem(m,q,0)[0]*special.mathieu_modcem1(m,q,z)[0]/\
        (np.sqrt(q)*special.mathieu_even_coef(m,q)[0]*(-1.0)**((m-1)/2+1))
    return cer

def Se(m,q,z): #Se definen las funciones de Mathieu modificadas a traves de las funciones de Mathieu de primer orden
    ser=0.0
    if m%2==0:
        ser=special.mathieu_sem(m,q,0)[1]*special.mathieu_sem(m,q,90)[1]*special.mathieu_modsem1(m,q,z)[0]/\
        (special.mathieu_odd_coef(m,q)[0]*q*(-1.0)**(m/2))
    else:
        ser=special.mathieu_sem(m,q,0)[1]*special.mathieu_sem(m,q,90)[0]*special.mathieu_modsem1(m,q,z)[0]/\
        (np.sqrt(q)*special.mathieu_odd_coef(m,q)[0]*(-1.0)**((m-1)/2))
    return ser

#******************************************************************************************
#Modificación hecha por Luis:    
def mascara_impar(m,h,q):
	mascara = np.zeros((600,800))
	
#	if m%2 == 0:
#		for i in range(600):
#			for j in range(800):
#				x=-8+0.02*j
#				y=-6+0.02*i
#				mascara[i][j] = special.mathieu_sem(m,q,ang_eta(x,y,h))[0]*Se(m,q,xi(x,y,h))
#				y=-6+0.02*i
#				mascara[i][j]=special.mathieu_sem(m,q,ang_eta(x,y,h))[0]*Se(m,q,xi(x,y,h))


	if m%2 == 0:
		for i in range(300):
			for j in range(400, 800):
				x=-8+0.02*j
				y=-6+0.02*i
				mascara[i][j] = special.mathieu_sem(m,q,ang_eta(x,y,h))[0]*Se(m,q,xi(x,y,h))
		mascara[300:600,0:400] = mascara[0:300,400:800][:,::-1][::-1]
		
		for i in range(300):
			for j in range(0, 400):
				x=-8+0.02*j
				y=-6+0.02*i
				mascara[i][j] = special.mathieu_sem(m,q,ang_eta(x,y,h))[0]*Se(m,q,xi(x,y,h))
		mascara[300:600,400:800] = mascara[0:300,0:400][:,::-1][::-1]	

	
	if m%2 != 0:
		for i in range(600):
			for j in range(400, 800):
				x=-8+0.02*j
		mascara[0:600,0:400] = mascara[0:600,400:800][:,::-1]
		
	return mascara          


#*******************************************************************************************

#Modificación hecha por Luis:    
def mascara_par(m,h,q):
	mascara = np.zeros((600,800))
	for i in range(300):
         for j in range(800):
              x=-8+0.02*j
              y=-6+0.02*i
              mascara[i][j]=special.mathieu_cem(m,q,ang_eta(x,y,h))[0]*Ce(m,q,xi(x,y,h))
	mascara[300:600,0:800]=mascara[0:300,0:800][::-1]
	
	return mascara        


#********************************************************************************************

def estructura_fase(matriz,dos_pi=CONST_2PI):
    nueva=np.zeros((600,800))
    for i in range(600):    # for every col:
      for j in range(800):    # For every row
        nueva[i,j] = round(math.fmod(cmath.phase(matriz[i][j]),2*np.pi)*dos_pi/(2*np.pi))
    return np.array(nueva, dtype=np.uint8)

def escala_gris(matriz,dos_pi=CONST_2PI):
    producto=1/np.max(-np.min(matriz)+matriz)
    matriz=np.around((-np.min(matriz)+matriz)*producto*dos_pi)
    return np.array(matriz,dtype=np.uint8)

def imagen_gris(matriz):
    gris=escala_gris(escala_gris(np.angle(matriz))+escala_gris(PATRON_CORRECTOR))
    return Image.fromarray(gris,'L')  

def coeficienteA(m,q):
    A=0.0
    if m%2==0:
        A=2*np.pi*special.mathieu_even_coef(m,q)[0]/(special.mathieu_cem(m,q,90)[0]*special.mathieu_cem(m,q,0)[0])
    else:
        A=-2*np.pi*np.sqrt(q)*special.mathieu_even_coef(m,q)[0]/(special.mathieu_cem(m,q,0)[0]*special.mathieu_cem(m,q,90)[1])
    return A
    
def coeficienteB(m,q):
    B=0.0
    if m%2==0:
        B=2*np.pi*q*special.mathieu_odd_coef(m,q)[0]/(special.mathieu_sem(m,q,90)[1]*special.mathieu_sem(m,q,0)[1])
    else:
        B=2*np.pi*np.sqrt(q)*special.mathieu_odd_coef(m,q)[0]/(special.mathieu_sem(m,q,90)[0]*special.mathieu_sem(m,q,0)[0])
    return B
  
def helicoidal(m,h,q):
    return coeficienteA(m,q)*mascara_par(m,h,q)  +  1j*coeficienteB(m,q)*mascara_impar(m,h,q)
    
    
    
def helicoidal_i(m,h,q):
    return coeficienteA(m,q)*mascara_par(m,h,q)-1j*coeficienteB(m,q)*mascara_impar(m,h,q)

def helicoidal_fase(m,h,q,fase):
	return coeficienteA(m,q)*mascara_par(m,h,q)+np.exp(1j*fase)*coeficienteB(m,q)*mascara_impar(m,h,q)

##################### Comienza el algoritmo de Gerchberg-Saxton ########################################################################

MF=np.linspace(-50/2, 50/2, 599, endpoint=False)
MF_original=np.linspace(-6,6,599, endpoint=False)


def luzreflejada(ideal,cintura):
    #Dado que estoy tomando como variable "ideal" a una matriz generada por las funciones mascara, y busco realizar una
    # fft primero debo de recortar la matriz. Para tener una matriz cuadrada, de lo contrario obtendere una matriz 
    # deformada
    #La cintura se toma en milimetros
    nluz=np.zeros((599,599),complex)
    for i in range(599):
      for j in range(599):
             x=-6+0.02*j
             y=-6+0.02*i
             r2=x**2+y**2 
             nluz[i,j]=np.exp(-r2/cintura**2)*np.exp(1j*cmath.phase(ideal[i,j+96]))   #Nota Luis: Esta es la forma del haz después de
																						#pasar por el SLM (ver Tesis Jorge  pag.27)
    return nluz

def aproximacioncft(matriz,tx=6,ty=6,Fx=50,Fy=50,largo=599,fx=MF,fy=MF):
  #La matriz es el conjunto de datos a los cuales se le pretende aplicar la transformada de Fourier
  # tx,ty son los puntos en el espacio a partir de donde se conoce la funcion $f$ que se pretende transformar
  #Fx,Fy es la frecuencia de muestro
  #La función aproximación se encuentra optimizada para arreglos de 599x599, y con frecuencia de muestro F=50
  paso1=np.fft.fft2(matriz)
  #fx=np.linspace(-Fx/2, Fx/2, largo, endpoint=False)
  #fy=np.linspace(-Fy/2, Fy/2, largo, endpoint=False)
  for i in range(largo):
     for j in range(largo):
        paso1[i,j]=paso1[i,j]*np.exp(2j*np.pi*fx[i]*tx)*np.exp(2j*np.pi*fy[j]*ty)*1./(Fx*Fy)
  return np.fft.fftshift(paso1)
  
def aproximacionicft(matriz,tx=25,ty=25,Fx=50/599,Fy=50/599,largo=599,fx=MF_original,fy=MF_original):
  paso1=np.fft.ifft2(matriz)
  for i in range(largo):
     for j in range(largo):
        paso1[i,j]=paso1[i,j]*np.exp(2j*np.pi*fx[i]*tx)*np.exp(2j*np.pi*fy[j]*ty)*1./(Fx*Fy)
  return np.fft.fftshift(paso1)
  
def remover_circulo(matriz,centro,radio):
    #Esta función sirve para colocar ceros en la región central, y eliminar la mancha del centro.
    #El centro se espera una arreglo [posicionx,posiciony], y el radio se espera en pixeles
    largoy,largox=matriz.shape
    for i in range(largoy):
        for j in range(largox):
            a=(i-centro[1])**2+(j-centro[0])**2
            if a<=radio**2:
                matriz[i,j]=0
    return matriz  
    
def reduccion(M1,M2):
    #Esta función toma como argumento dos matrices que representan una misma imagen, dim(M1)=nxn y dim(M2)=mxm. Con n<m
    #La función da como resultado una matriz M3 de dimensiones nxn la cual se obtiene de interpolar M2. Los puntos
    # que representan M1 y M3 son los mismos. El parametro escala, nos dice el espaciamiento entre puntos en la matriz M1.
    n=M1.shape[0]
    m=M2.shape[0]
    M3=np.zeros((n,n))
    for i in range(n-1):
            meq=m*i/n
            posv=[int(meq),int(meq)+1]
            difv=meq-int(meq)  
            for j in range(n-1):
                meqh=m*j/n
                posh=[int(meqh),int(meqh)+1]
                difh=meqh-int(meqh)
                M3[i,j]=((1-difh)*(1-difv)*M2[posv[0],posh[1]]+difh*(1-difv)*M2[posv[0],posh[1]]+
                difv*difh*M2[posv[0],posh[1]]+(1-difh)*difv*M2[posv[1],posh[0]])
    #Me falta una sección en forma de L reflejada
    for i in range(n-1):
        meq=m*i/n
        posv=int(meq)
        difh=meq-int(meq) 
        M3[n-1,i]=(1-difh)*M2[m-1,posv]+difh*M2[m-1,posv+1]
    for i in range(n-1):
        meq=m*i/n
        posv=int(meq)
        difv=meq-int(meq) 
        M3[i,n-1]=(1-difv)*M2[posv,m-1]+difv*M2[posv+1,m-1]
    M3[n-1,n-1]=M2[m-1,m-1]
    return M3
    
def FrFT_2d(matriz,alpha1,alpha2):
    m1,m2=matriz.shape
    y=np.zeros((2*m1,2*m2),complex)
    z=np.zeros((2*m1,2*m2),complex)
    y[:m1,:m2]=(np.exp(-1j*np.pi*alpha1*np.power(np.array(range(m1)),2).reshape(m1,1))*
                  np.exp(-1j*np.pi*alpha2*np.power(np.array(range(m2)),2))*matriz)
    z[:m1,:m2]=(np.exp(1j*np.pi*alpha1*np.power(np.array(range(m1)),2).reshape(m1,1))*
                  np.exp(1j*np.pi*alpha2*np.power(np.array(range(m2)),2)))
    z[m1:,:m2]=(np.exp(1j*np.pi*alpha1*np.power((np.array(range(m1,2*m1))-2*m1),2).reshape(m1,1))*
                  np.exp(1j*np.pi*alpha2*np.power(np.array(range(m2)),2)))
    z[:m1,m2:]=(np.exp(1j*np.pi*alpha1*np.power((np.array(range(m1))),2).reshape(m1,1))*
                  np.exp(1j*np.pi*alpha2*np.power((np.array(range(m2,2*m2))-2*m2),2)))
    z[m1:,m2:]=(np.exp(1j*np.pi*alpha1*np.power((np.array(range(m1,2*m1))-2*m1),2).reshape(m1,1))*
                  np.exp(1j*np.pi*alpha2*np.power((np.array(range(m2,2*m2))-2*m2),2)))
    ny=np.fft.fft2(y)
    nz=np.fft.fft2(z)
    exponente=(np.exp(-1j*np.pi*alpha1*np.power(np.array(range(2*m1)),2).reshape(2*m1,1))*
                  np.exp(-1j*np.pi*alpha2*np.power(np.array(range(2*m2)),2)))
    return exponente*np.fft.ifft2(ny*nz)

def approx_frft2dm(matriz,a1,a2,gamma1,gamma2):
    m1,m2=matriz.shape
    beta1=a1/m1
    beta2=a2/m2
    delta1=beta1*gamma1
    delta2=beta2*gamma2
    nueva_matriz=(np.exp(1j*np.pi*delta1*m1*np.array(range(m1)).reshape(m1,1))*
                  np.exp(1j*np.pi*delta2*m2*np.array(range(m2)))*matriz)
    nuevo_arreglo=FrFT_2d(nueva_matriz,delta1,delta2)[:m1,:m2]
    exponente=beta1*beta2*(np.exp(1j*np.pi*(np.array(range(m1)).reshape(m1,1)-m1/2)*m1*delta1)*
                  np.exp(1j*np.pi*(np.array(range(m2))-m2/2)*m2*delta2))
    return nuevo_arreglo*exponente

def error_amp_r(simulacion,foto,normalizacion):
    N=600*800
    #El error se encuentra de forma porcentual
    
    return (np.sum(np.square(np.abs(simulacion)/np.max(np.abs(simulacion))-np.sqrt(foto/normalizacion))))*100/N
    
def inv_frft2d(matriz,f=300,londa=776e-6,my=600,mx=800,lorigy=12,lorigx=16,res=0.0052):
    #La funcion esta optimizada para realizar transformada inversas de anillos al espacio original
    ax=mx*res/(f*londa)
    ay=my*res/(f*londa)
    gammax=lorigx/mx
    gammay=lorigy/my
    return  np.flipud(np.fliplr(approx_frft2dm(matriz,ay,ax,gammay,gammax)))
# Aparentemente esta al reves la posición en xy, pero eso se debe a la forma en la cual se lee una matriz.

def luzreflejada_c(ideal,cintura):
    #Voy a realizar la transformación de la matriz rectangular. El objetivo es que funcione con el algoritmo de correcion
    nluz=np.zeros((600,800),complex)
    for i in range(600):
      for j in range(800):
             x=-8+0.02*j
             y=6-0.02*i
             r2=x**2+y**2 
             nluz[i,j]=np.exp(-r2/cintura**2)*np.exp(1j*cmath.phase(ideal[i,j]))
    return nluz
    

def correccion_imagen(fase,error,parametros,f=300,londa=776e-6):
    #parametros es un arreglo que se espera tenga el orden:
    #[centro, radio, potencia, kt,cintura,factor_conversion,error] 
    centro=parametros[0]
    radio=parametros[1]
    kt=parametros[2]
    cintura=parametros[3]
    limite=2 #Establezco un numero maximo de iteraciones. Dado que toma alrededor de 2 minutos generar una imagen, el 
    #algoritmo solo se iterara 10 veces. Para que así no demore mas de media hora la preparación de la luz.
    #Se envia la imagen al SLM
    #marco=fase+0.0 #Copia la matriz pero en otra ubicacion
    fase=fase.astype(complex)
    monitor2(grayImageCorr(estructura_fase(fase)))
    sleep(2)
    call(['./frametest'])
    #Ya se tomo la primer foto, previamente se debio de haber tomado una fotografia con la misma mascara de fase aqui 
    #usada. El objetivo de esta foto es encontrar el centro y el radio.
    error0=100 # Se inicializa con un error muy alto, con el fin de al menos realizar un ciclo.
    n=0  #n es un contador
    haz_gauss_or=luzreflejada_c(fase,cintura)
    resolucion_camara=5.2e-3
    radio_anillo=resolucion_camara/(f*londa)
    anillo_sim_or=(1/1j*londa*f)*approx_frft2dm(haz_gauss_or,12,16,radio_anillo,radio_anillo)    #Nota Luis: Al correr esta línea se 
                                                                                                # obtiene la luz después de pasar por el 
                                                                                                # primer lente (ver Tesis Jorge pag.27).      
    anillo_foto=np.array(Image.open("snap_BGR8.png").rotate(90)) #Creo que la rotacion va a ser innecesaria
    anillo_foto=remover_circulo(anillo_foto,centro,radio)
    normalizar=np.max(anillo_foto)
    while error0>=error:
        n=n+1
        if n==limite:
            break
        #print(fase.shape)
        haz_gauss=luzreflejada_c(fase,cintura) #Aqui tal vez se introduce error
        anillo_sim=(1/1j*londa*f)*approx_frft2dm(haz_gauss,12,16,radio_anillo,radio_anillo)
        anillo_foto=np.array(Image.open("snap_BGR8.png").rotate(90))
        anillo_foto=remover_circulo(anillo_foto,centro,radio) #Lo quiero quitar
        foto_comp=anillo_foto[(centro[1]-300):(centro[1]+300),(centro[0]-400):(centro[0]+400)]
        error0=error_amp_r(anillo_sim_or,foto_comp,normalizar)
        print("El error es"+" "+str(error0))
        n_haz=np.sqrt(foto_comp)*np.exp(1j*np.angle(anillo_sim))
        haz_regreso=inv_frft2d(n_haz)
        fase=np.exp(1j*np.angle(haz_regreso))
        #fase=np.exp(1j*np.angle(inv_frft2d(n_haz)))
        monitor2(grayImageCorr(estructura_fase(haz_regreso)))
        #monitor2(grayImageCorr(fase))
        sleep(2)
        call(['./frametest']) 
    return     
#####################################Haces Bessel ##################################################################
#def mascara_Bessel(kperp,m,lambd=776e-6):
    #k=2*np.pi/lambd
    #kz=np.sqrt(k**2-kperp**2)
    #mascara=np.zeros((600,800),complex)
    #for i in range(600):
     #   for j in range(800):
      #        x=-8+0.02*j
       #       y=-6+0.02*i
        #      r=np.sqrt(x**2+y**2)
        #      theta=np.arctan2(y,x)
         #     mascara[i][j]=np.exp(1j*m*theta)*special.jv(m,kperp*r)
    #return mascara
    
def mascara_Bessel(kperp,m):
    #lambd=776e-6
    #k=2*np.pi/lambd     
    #kz=np.sqrt(k**2-kperp**2)
    
    mascara=np.zeros((600,800),complex)   #Define un arreglo de 600 renglones y 800 columnas
    
    for i in range(600):
        for j in range(800):
            x = -8 + 0.02*j     # 0.2 mm es el tamaño del pixel del SLM
            y = -6 + 0.02*i     # Los valores -2 y -1.5 estan dados en milimetros 
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(y,x)
            mascara[i][j] = -2*np.pi*((1j)**m)*kperp*np.exp(1j*m*theta)*special.jv(m,kperp*r)  # "special.jv" es la función Bessel
            
    return mascara
    
######################################################Haces Parabolicos ####################################################33
def u_parabolico(x,y):
    u=0.
    r=np.sqrt(x**2+y**2)
    if y>=0:
        u=np.sqrt(x+r)
    else:
        u=-np.sqrt(x+r)
    return u
  
def mascara_parabolica(m,kperp,a,paridad):
    mascara=np.zeros((600,800),complex)
    if paridad=="p":
       for i in range(600):
           for j in range(800):
              x=-8+0.02*j
              y=-6+0.02*i
              u=u_parabolico(x,y)
              v=sqrt(-x+hypot(x,y))
              hyper=mp.hyp1f1(0.25-0.5*a*1j,0.5,1j*kperp*u**2)*mp.hyp1f1(0.25+0.5*a*1j,0.5,1j*kperp*u**2)
              hyper_r=float(hyper.real)
              hyper_i=float(hyper.imag)
              hyper_np=hyper_r+hyper_r*1j
              mascara[i][j]=np.exp(-1j*kperp*0.5*(u**2+v**2))*hyper_np
    elif paridad=="i":
        for i in range(600):
           for j in range(800):
              x=-8+0.02*j
              y=-6+0.02*i
              hyper=mp.hyp1f1(0.75-0.5*a*1j,1.5,1j*kperp*u**2)*mp.hyp1f1(0.75+0.5*a*1j,1.5,1j*kperp*u**2)
              hyper_r=float(hyper.real)
              hyper_i=float(hyper.imag)
              hyper_np=hyper_r+hyper_r*1j
              mascara[i][j]=np.exp(-1j*kperp*0.5*(u**2+v**2))*hyper_np
    else:
        print("La paridad esta mal escrita,escriba p para par ó i para impar")
    return mascara   
    
##################################################### FIN DE Haces Parabolicos #################################################   
    
############################################################################################################################    
############################  Arreglos de vortices    ######################################################################
############################################################################################################################


def principal_mask_vortices():
    #gray2pi = gray2pi + 1  #149    #%gray level defining 2*pi phase
    gray0 = 0
    Xpix = 800      # %number of pixels of the SLM along x
    Ypix = 600      # %number of pixels of the SLM along y

    xx = np.arange(-(Xpix / 2), (Xpix / 2), 1)
    yy = np.arange(-(Ypix / 2), (Ypix / 2), 1)
    X, Y = np.meshgrid(xx, yy)
    
    Nx = int(input('Numero de vortices en direccion X: '))
    Ny = 1   # %input('Numero de vortices en direccion Y: ');


    if Nx != 1:
        Px = int(input('Separación de los vórtices en X en píxeles: '))
    else:
        Px = 0

    if Ny != 1:
        Py = int(input('Separación de los vórtices en Y en píxeles: '))
    else:
        Py = 0


    top=1   #%input('Elegir 1 para vortices de igual carga y 2 para cargas anticorrelacionadas: ');

    l=1   # %topological charge when top=1

    
    PhaseMask = Fun_vortexlattice(X,Y,Nx,Ny,Px,Py,top,l)
    
    #PhaseMask = PhaseMask + 2*np.pi
    PhaseMask = np.mod(PhaseMask + 2 * np.pi, 2 * np.pi)
    #gray = ((PhaseMask * (gray2pi - gray0) / (2 * np.pi)) + gray0).astype(np.uint8)
    
    
    return PhaseMask


def Fun_vortexlattice(X, Y, Nx, Ny, Px, Py, top, l):
    #%script for generating the vortex phase function
    #%[X,Y] are the SLM pixels
    #%Nx, Ny is the number of vortices along X and Y
    #%Px, Py separation between adjacent vortices in pixels along X and Y

    #%top=1 for all the vortices of the same charge (+/-1)
    #%top=2 for anticorrelated-sing of nearest neighbors

    #%l=topological charge when top=1

    ###  %Defining the vortex positions within an ordered  lattice:

    if Nx == 1:
        NN = np.array([0])
    elif Nx % 2 == 0:
        NN0 = round(Nx - 1) * Px / 2
        NN = np.arange(-NN0, -NN0 + (Nx - 1) * Px + Px, Px)
        NN = np.floor(NN).astype(int)
    else:
        NN = np.arange(-((Nx - 1) / 2) * Px, ((Nx - 1) / 2) * Px + Px, Px)
        NN = np.round(NN).astype(int)


    if Ny == 1:
        MM = np.array([0])
    elif Ny % 2 == 0:
        MM0 = round(Ny - 1) * Py / 2
        MM = np.arange(-MM0, -MM0 + (Ny - 1) * Py + Py, Py)
        MM = np.floor(MM).astype(int)
    else:
        MM = np.arange(-((Ny - 1) / 2) * Py, ((Ny - 1) / 2) * Py + Py, Py)
        MM = np.round(MM).astype(int)

    LL = l * np.ones((Nx, Ny))
    
    if top == 2:
        for j in range(Ny):
            if j % 2 == 1:  # j in MATLAB starts from 1, so adjust for Python's 0-based index
                LL[:, j] = -1

        for i in range(Nx):
            if i % 2 == 1:  # i in MATLAB starts from 1, so adjust for Python's 0-based index
                LL[i, :] = -LL[i, :]

    fasor = 1

    for j in range(Ny): 
        for i in range(Nx):
            fase = LL[i, j] * np.arctan2(Y - MM[j], X - NN[i])
            fasor *= np.exp(1j * fase)

    fasetotal = np.angle(fasor)

    return fasetotal



def U(X,Y,l,alpha):
    resultado= special.jv(0, alpha*np.sqrt(X**2+Y**2)) * np.sin(l*np.arctan2(Y,X))  
    return resultado

def blanco():


    #cord1 = np.arange(-300,300,1, dtype=float)
    #cord2 = np.arange(-400,400,1, dtype=float)
    #X, Y = np.meshgrid(cord1, cord2, indexing = "ij")
     
    mascara=np.zeros((600,800),complex)

    #gray2pi = 149    #%gray level defining 2*pi phase
    #gray0 = 0
    Xpix = 800      # %number of pixels of the SLM along x
    Ypix = 600 
    xx = np.arange(-(Xpix / 2), (Xpix / 2), 1)
    yy = np.arange(-(Ypix / 2), (Ypix / 2), 1)
    X, Y = np.meshgrid(xx, yy)

    Pi=np.pi
    g2pi=149 # 221,148;
    gray2pi = (2.*Pi)/g2pi
    pi2gray = g2pi/(2.*Pi)
    D=800.  #apertura efectiva en el SLM
    l=2.    #2*l es el numero de secciones acimutales
    m=3.    #numero aproximado de circulos concentricos
    alpha=m*2.4/(D/2)
    
    rr = np.sqrt(X**2 + Y**2)
    circle = (rr <= round(D / 2)).astype(np.float64)

    #blanco = np.angle(Bessel0) + np.pi
    #blanco = (pi2gray * np.mod(blanco + corr, 2 * np.pi) * circle).astype(np.uint8)

    
    Un = U(X,Y,l,alpha)
    blanco = np.angle(Un)+2*Pi
    blanco2 = pi2gray *np.mod(blanco, 2 * np.pi)*circle#%(3*Pi)  #Factor de 3Pi para un mejor contraste
    #blanco2 = pi2gray *(blanco)*circle%(3*Pi)
        # %number of pixels of the SLM along y
     



    #PhaseMask = blanco2
    #PhaseMask = np.mod(PhaseMask + 2 * np.pi, 2 * np.pi)

    #mascara = ((PhaseMask * (gray2pi - gray0) / (2 * np.pi)) + gray0).astype(np.uint8)
    #type(mascara), mascara.dtype

    
    return blanco2 

def vortex_matlab():
    pathhh = '/home/atomos_frios/LAFriOC/projects/propagation_of_vortices/scripts/mascara_fase_matlab/'
    mat = scipy.io.loadmat(pathhh+"blanco1.mat")
    data=mat["blanco"]
    return data

# Funciones para agregar la imagen de corrección *********************************************
#*************************************************************************
direct_3 = '/home/atomos_frios/LAFriOC/projects/Atomoscalientes/docs/SLMdisco/deformation_correction_pattern/' 
list_archvs = []
for archvs in listdir(direct_3):
    #direct3 = direct_3 +archvs
    list_archvs.append(archvs)
    end_list = sorted(list_archvs)
    
im = np.array(Image.open(direct_3+end_list[29]))
im_corr = np.array(np.concatenate( (im,np.zeros(shape=(600,8))) ,axis=1 )   ,dtype=np.uint8)
im_corr_ang = im_corr*2*np.pi/CONST_2PI

def escala_gris3(matriz,dos_pi=CONST_2PI):
    matriz = matriz+im_corr_ang
    matriz = np.around(np.fmod(matriz, 2*np.pi )*dos_pi/(2*np.pi) )
    return np.array(matriz,dtype=np.uint8)

def escala_gris2(matriz,dos_pi=CONST_2PI):
    matriz2 = matriz+im_corr_ang
    norm = 1/np.max(-np.min(matriz2)+matriz2)
    matriz2 = np.around((-np.min(matriz2)+matriz2)*norm*dos_pi)
    return np.array(matriz2,dtype=np.uint8)

def escala_gris4(matriz,dos_pi=CONST_2PI):
    matriz = np.around((np.fmod(matriz, 2*np.pi )+im_corr_ang)*dos_pi/(2*np.pi) )
    return np.array(matriz,dtype=np.uint8)
