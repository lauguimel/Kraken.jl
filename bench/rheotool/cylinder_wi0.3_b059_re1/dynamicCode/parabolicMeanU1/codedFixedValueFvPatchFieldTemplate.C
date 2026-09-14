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
    // SHA1 = a25f6bf266688079c452d1d88fd3dc60f3f12bf6
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void parabolicMeanU1_a25f6bf266688079c452d1d88fd3dc60f3f12bf6(bool load)
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
    parabolicMeanU1FixedValueFvPatchVectorField
);


const char* const parabolicMeanU1FixedValueFvPatchVectorField::SHA1sum =
    "a25f6bf266688079c452d1d88fd3dc60f3f12bf6";


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

parabolicMeanU1FixedValueFvPatchVectorField::
parabolicMeanU1FixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF)
{
    if (false)
    {
        Info<<"construct parabolicMeanU1 sha1: a25f6bf266688079c452d1d88fd3dc60f3f12bf6"
            " from patch/DimensionedField\n";
    }
}


parabolicMeanU1FixedValueFvPatchVectorField::
parabolicMeanU1FixedValueFvPatchVectorField
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
        Info<<"construct parabolicMeanU1 sha1: a25f6bf266688079c452d1d88fd3dc60f3f12bf6"
            " from patch/dictionary\n";
    }
}


parabolicMeanU1FixedValueFvPatchVectorField::
parabolicMeanU1FixedValueFvPatchVectorField
(
    const parabolicMeanU1FixedValueFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper)
{
    if (false)
    {
        Info<<"construct parabolicMeanU1 sha1: a25f6bf266688079c452d1d88fd3dc60f3f12bf6"
            " from patch/DimensionedField/mapper\n";
    }
}


parabolicMeanU1FixedValueFvPatchVectorField::
parabolicMeanU1FixedValueFvPatchVectorField
(
    const parabolicMeanU1FixedValueFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF)
{
    if (false)
    {
        Info<<"construct parabolicMeanU1 sha1: a25f6bf266688079c452d1d88fd3dc60f3f12bf6 "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

parabolicMeanU1FixedValueFvPatchVectorField::
~parabolicMeanU1FixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy parabolicMeanU1 sha1: a25f6bf266688079c452d1d88fd3dc60f3f12bf6\n";
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void parabolicMeanU1FixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        Info<<"updateCoeffs parabolicMeanU1 sha1: a25f6bf266688079c452d1d88fd3dc60f3f12bf6\n";
    }

//{{{ begin code
    #line 29 "//data/0/U/boundaryField/inlet"
const vectorField& Cf = patch().Cf();
            vectorField Uin(patch().size(), vector::zero);

            const scalar Umean = 1.0;
            const scalar halfHeight = 2.0;

            forAll(Uin, faceI)
            {
                const scalar y = Cf[faceI].y();
                const scalar profile = 1.5*Umean*(1.0 - sqr(y/halfHeight));
                Uin[faceI] = vector(max(profile, scalar(0)), 0, 0);
            }

            operator==(Uin);
//}}} end code

    this->fixedValueFvPatchField<vector>::updateCoeffs();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

