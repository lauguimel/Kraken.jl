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

defineTypeNameAndDebug(kineticEnergyFunctionObject, 0);

addRemovableToRunTimeSelectionTable
(
    functionObject,
    kineticEnergyFunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = 43814293f77940407d31fed82bd2846394aaa497
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void kineticEnergy_43814293f77940407d31fed82bd2846394aaa497(bool load)
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

const fvMesh& kineticEnergyFunctionObject::mesh() const
{
    return refCast<const fvMesh>(obr_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

kineticEnergyFunctionObject::kineticEnergyFunctionObject
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

kineticEnergyFunctionObject::~kineticEnergyFunctionObject()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool kineticEnergyFunctionObject::read(const dictionary& dict)
{
    if (false)
    {
        Info<<"read kineticEnergy sha1: 43814293f77940407d31fed82bd2846394aaa497\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool kineticEnergyFunctionObject::execute()
{
    if (false)
    {
        Info<<"execute kineticEnergy sha1: 43814293f77940407d31fed82bd2846394aaa497\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool kineticEnergyFunctionObject::write()
{
    if (false)
    {
        Info<<"write kineticEnergy sha1: 43814293f77940407d31fed82bd2846394aaa497\n";
    }

//{{{ begin code
    #line 68 "//data/system/controlDict/functions/kineticEnergy"
// Lookup/create variables 
       
           const volVectorField& U = mesh().lookupObject<volVectorField>("U");
           const volSymmTensorField& tau = mesh().lookupObject<volSymmTensorField>("tau");
           const dictionary& cttP = mesh().lookupObject<IOdictionary>("constitutiveProperties");
           dimensionedScalar lambda_(cttP.subDict("parameters").lookup("lambda"));
           dimensionedScalar etaP_(cttP.subDict("parameters").lookup("etaP"));

          // Compute kinetic energy

           int nCells = mesh().nCells(); 
           reduce(nCells, sumOp<int>());

           scalarList list;
           list.append(mesh().time().value()); // Time (col 0)  
           list.append( ( 0.5/nCells ) * gSum( mag( U.primitiveField() ) * mag( U.primitiveField() ) ) ); // Average kinE (col 1) 
           list.append( ( 0.5/nCells ) * (lambda_.value()/etaP_.value()) * gSum( tr( tau.primitiveField() ) ) ); // Average elastic energy (col 2)  
           
          // Write data

           string comsh;           
           string filename("kinEner.txt");
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


bool kineticEnergyFunctionObject::end()
{
    if (false)
    {
        Info<<"end kineticEnergy sha1: 43814293f77940407d31fed82bd2846394aaa497\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

