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

defineTypeNameAndDebug(outputCdFunctionObject, 0);

addRemovableToRunTimeSelectionTable
(
    functionObject,
    outputCdFunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = 8ab66f9aaf8e23a872e706b9c7d002d0a00278d2
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void outputCd_8ab66f9aaf8e23a872e706b9c7d002d0a00278d2(bool load)
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

const fvMesh& outputCdFunctionObject::mesh() const
{
    return refCast<const fvMesh>(obr_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

outputCdFunctionObject::outputCdFunctionObject
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

outputCdFunctionObject::~outputCdFunctionObject()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool outputCdFunctionObject::read(const dictionary& dict)
{
    if (false)
    {
        Info<<"read outputCd sha1: 8ab66f9aaf8e23a872e706b9c7d002d0a00278d2\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool outputCdFunctionObject::execute()
{
    if (false)
    {
        Info<<"execute outputCd sha1: 8ab66f9aaf8e23a872e706b9c7d002d0a00278d2\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool outputCdFunctionObject::write()
{
    if (false)
    {
        Info<<"write outputCd sha1: 8ab66f9aaf8e23a872e706b9c7d002d0a00278d2\n";
    }

//{{{ begin code
    #line 67 "//data/system/controlDict/functions/outputCd"
// Lookup/create variable 
	     
	   const volVectorField& U = mesh().lookupObject<volVectorField>("U");
           const volSymmTensorField& tau = mesh().lookupObject<volSymmTensorField>("tau");
           const volScalarField& p = mesh().lookupObject<volScalarField>("p");
           const dictionary& constDict = mesh().lookupObject<IOdictionary>("constitutiveProperties");
           dimensionedScalar rho_(constDict.subDict("parameters").lookup("rho"));
           dimensionedScalar etaS_(constDict.subDict("parameters").lookup("etaS"));
           dimensionedScalar etaP_(constDict.subDict("parameters").lookup("etaP"));
  
           label cyl = mesh().boundaryMesh().findPatchID("cylinder");
           scalarList list;
    
          // Compute cd
 
           volTensorField L(fvc::grad(U));

	   volSymmTensorField F(tau + symm( L + L.T() ) * etaS_ - p * symmTensor::I * rho_);

           vector Fpatch = gSum( ( -mesh().boundaryMesh()[cyl].faceAreas() ) & F.boundaryField()[cyl] )/(etaS_ + etaP_).value();
  
           list.append(mesh().time().value());  // Time (col 0)  
           list.append(Fpatch.x());             // Cd   (col 1)  

          // Write data

           string comsh;           
           string filename("Cd.txt");
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


bool outputCdFunctionObject::end()
{
    if (false)
    {
        Info<<"end outputCd sha1: 8ab66f9aaf8e23a872e706b9c7d002d0a00278d2\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

