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

#include "codedFixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = 128daf6f72aae42ab289e0cdd052029347c02b2c
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void smoothU_128daf6f72aae42ab289e0cdd052029347c02b2c(bool load)
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

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchVectorField,
    smoothUFixedValueFvPatchVectorField
);


const char* const smoothUFixedValueFvPatchVectorField::SHA1sum =
    "128daf6f72aae42ab289e0cdd052029347c02b2c";


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

smoothUFixedValueFvPatchVectorField::
smoothUFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF)
{
    if (false)
    {
        Info<<"construct smoothU sha1: 128daf6f72aae42ab289e0cdd052029347c02b2c"
            " from patch/DimensionedField\n";
    }
}


smoothUFixedValueFvPatchVectorField::
smoothUFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict)
{
    if (false)
    {
        Info<<"construct smoothU sha1: 128daf6f72aae42ab289e0cdd052029347c02b2c"
            " from patch/dictionary\n";
    }
}


smoothUFixedValueFvPatchVectorField::
smoothUFixedValueFvPatchVectorField
(
    const smoothUFixedValueFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper)
{
    if (false)
    {
        Info<<"construct smoothU sha1: 128daf6f72aae42ab289e0cdd052029347c02b2c"
            " from patch/DimensionedField/mapper\n";
    }
}


smoothUFixedValueFvPatchVectorField::
smoothUFixedValueFvPatchVectorField
(
    const smoothUFixedValueFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF)
{
    if (false)
    {
        Info<<"construct smoothU sha1: 128daf6f72aae42ab289e0cdd052029347c02b2c "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

smoothUFixedValueFvPatchVectorField::
~smoothUFixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy smoothU sha1: 128daf6f72aae42ab289e0cdd052029347c02b2c\n";
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void smoothUFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        Info<<"updateCoeffs smoothU sha1: 128daf6f72aae42ab289e0cdd052029347c02b2c\n";
    }

//{{{ begin code
    #line 30 "//data/0/U/boundaryField/inlet"
const scalar& t = this->db().time().timeOutputValue();
 
          vector Uav(0.25, 0, 0);
          vector dirN(1, 0, 0);
          
          scalar tlim(1.);
          scalar fac(8.);

          if (t<=tlim)
           {
             Uav = ( ( (1 - Foam::cos( 3.1415926535897932 * t/tlim) ) / fac) * dirN);
           }
       
          operator == (Uav);
//}}} end code

    this->fixedValueFvPatchField<vector>::updateCoeffs();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

