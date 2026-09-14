/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) YEAR OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "codedFunctionObjectTemplate.H"
#include "fvCFD.H"
#include "unitConversion.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(codeWriteFunctionObject, 0);

addRemovableToRunTimeSelectionTable
(
    functionObject,
    codeWriteFunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = b90088fdf4a718dc84b608e9d00fa903ee9c02bf
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void codeWrite_b90088fdf4a718dc84b608e9d00fa903ee9c02bf(bool load)
    {
        if (load)
        {
            // code that can be explicitly executed after loading
        }
        else
        {
            // code that can be explicitly executed before unloading
        }
    }
}


// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

const fvMesh& codeWriteFunctionObject::mesh() const
{
    return refCast<const fvMesh>(obr_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

codeWriteFunctionObject::codeWriteFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    functionObjects::regionFunctionObject(name, runTime, dict)
{
    read(dict);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

codeWriteFunctionObject::~codeWriteFunctionObject()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool codeWriteFunctionObject::read(const dictionary& dict)
{
    if (false)
    {
        Info<<"read codeWrite sha1: b90088fdf4a718dc84b608e9d00fa903ee9c02bf\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool codeWriteFunctionObject::execute()
{
    if (false)
    {
        Info<<"execute codeWrite sha1: b90088fdf4a718dc84b608e9d00fa903ee9c02bf\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool codeWriteFunctionObject::write()
{
    if (false)
    {
        Info<<"write codeWrite sha1: b90088fdf4a718dc84b608e9d00fa903ee9c02bf\n";
    }

//{{{ begin code
    #line 182 "//data/system/controlDict/functions/codeWrite"
// Lookup/create variable 

           label patchvort = mesh().boundaryMesh().findPatchID("wall_liptop"); // Define name of BC contacting the vortex
           const polyPatch& cPatchvort = mesh().boundaryMesh()[patchvort];
           const volVectorField& U = mesh().lookupObject<volVectorField>("U");
           const volVectorField& C = mesh().C();

          // Define reference parameters 
 
           vector refPoint(0., 1., 0.5); // Reference point to zero the vortex length
           vector refDir(0., 1., 0.); // Vector aligned with the wall

          // Compute vortex length based on the point of velocity inversion

           scalarList list;
           list.append(mesh().time().value()); // Time (col 0)  
           int index(0);
           scalar uPrev=0.0; vector CPrev(0., 0., 0.);
           vector refDirU(refDir/mag(refDir));
      
           forAll(cPatchvort, facei )       
              {
                label  curCell = cPatchvort.faceCells()[facei];
                scalar uCmp = (U[curCell] & refDirU);
               
                if (uPrev*uCmp<0.0)
                 {
                   vector r_curCell = -uCmp * ( CPrev - C[curCell] ) / (uPrev - uCmp) + C[curCell];
  
                   list.append( mag( ( (r_curCell - refPoint) & refDirU ) ) ); // Distance between refPoint and inversion points (col 1:n)

                   index++;
                 } 
       
                uPrev = uCmp;
                CPrev = C[curCell];
            }  
             
          // Write data

           string comsh;           
           string filename("Lip_top.txt");
	   std::stringstream doub2str; doub2str.precision(12);

           comsh = "./writeData " + filename;
           forAll(list, id)
            {
              doub2str.str(std::string());
              doub2str << list[id]; 
              comsh += " " + doub2str.str();
            }
           
	    if (Pstream::master())
            {
	      system(comsh);
            }
//}}} end code

    return true;
}


bool codeWriteFunctionObject::end()
{
    if (false)
    {
        Info<<"end codeWrite sha1: b90088fdf4a718dc84b608e9d00fa903ee9c02bf\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

